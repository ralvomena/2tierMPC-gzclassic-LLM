#!/usr/bin/env python3

import rclpy
import math
import psutil
import rclpy.qos as qos
import numpy as np
import threading as th
from local_tier_pkg.mpc_tracking_core import MPC
from functools import partial
from rclpy.node import Node

# Messages and services interfaces
from interfaces_pkg.msg import Trajectory, AGVList, SimStatus
from interfaces_pkg.srv import CallMPC, CloseMPC
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# Thread lock
lock = th.Lock()

class MPCLocalNode(Node, MPC):
    """
    MPC tracking node class. This class sets up the ROS environment for MPC. 
    It also saves the simulation data, if enabled.
    """

    def __init__(self):
        """
        MPCLocalNode constructor.
        """
        # Inherit Node class and pass a default node name (can be changed by a parameter in launch file)
        super().__init__("agv_1_mpc_local_tracking")

        # Inherit MPC class passing the parameters
        super(Node, self).__init__(N=60, T=0.1, Q=np.diag([10, 10, 20]), R=np.diag([10, 20]), ma57_solver=False)

        # Declare and get parameters (all of them can be changed in launch file)
        self.declare_parameter("agv_id", "agv_1")
        self.agv_id = self.get_parameter("agv_id").value
        self.declare_parameter("priority", "medium")
        self.priority = self.get_parameter("priority").value
        self.declare_parameter("sim_name", "sim_name")
        self.sim_name = self.get_parameter("sim_name").value
        self.declare_parameter("save_sim_data", False)
        self.save_sim_data = self.get_parameter("save_sim_data").value
        self.declare_parameter("high_vel", 1.0)
        self.high_vel = self.get_parameter("high_vel").value
        self.declare_parameter("medium_vel", 0.75)
        self.medium_vel = self.get_parameter("medium_vel").value
        self.declare_parameter("low_vel", 0.75)
        self.low_vel = self.get_parameter("low_vel").value
        self.declare_parameter("scenario", 1)
        self.scenario = self.get_parameter("scenario").value
        self.declare_parameter("scn_size", [-100, -100, 100, 100]) #-x, -y, +x, +y
        self.scn_size = self.get_parameter("scn_size").value

        # Save the simulation data in this folder
        self.path = '/media/psf/Dropbox/Sim'

        # Open simulation data files
        if self.save_sim_data:
            self.trajectory_file = open(self.path + '/trajectory_' + self.agv_id + '_' + self.sim_name + '.txt', 'w')
            self.control_signal_file =  open(self.path +'/control_signal_' + self.agv_id + '_' + self.sim_name + '.txt', 'w')
            self.cpu_mem_file = open(self.path +'/cpu_mem_' + self.agv_id + '_' + self.sim_name + '_local.txt', 'w')
            self.mpc_time_file = open(self.path +'/mpc_time_' + self.agv_id + '_' + self.sim_name + '_local.txt', 'w')

        # Show terminal info on node launch
        self.get_logger().info(f"The local MPC node for {self.agv_id.upper()} has been started.")
        if self.save_sim_data:
            self.get_logger().info(f"Simulation data will be saved in: {self.path}, in files ending with: '{self.sim_name}'.")

        # Init some variables
        self.x0_start = False
        self.trajectory = np.zeros((self.N+1, 3))
        self.control_signal = []
        self.first_run = True
        self.process = psutil.Process()
        self.cpu_mem = []
        self.sim_status = False

        # Set QoS profile
        self.qos_profile = qos.QoSProfile(reliability=qos.QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT, 
            history=qos.QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, depth=0)

        # Start publishers and subscribers
        self.velocity_publisher = self.create_publisher(Twist, self.agv_id + '/cmd_vel', 10)
        self.odometry_publisher = self.create_publisher(Odometry, self.agv_id + '/odom/local', qos.qos_profile_sensor_data)
        self.odometry_subscriber = self.create_subscription(Odometry, self.agv_id + '/odom', self.callback_odometry, qos.qos_profile_sensor_data)
        self.trajectory_subscriber = self.create_subscription(Trajectory, self.agv_id + '/trajectory', self.callback_trajectory, self.qos_profile)
        self.agv_list_subscriber = self.create_subscription(AGVList, 'agv_list', self.callback_agv_list, 10)
        self.sim_status_subscriber = self.create_subscription(SimStatus, 'sim_status', self.callback_sim_status, 10) # Disable on Scenario 1 simulations

        # Start services
        self.mpc_tracking_service_name = self.agv_id + '/mpc/tracking'
        self.call_mpc_local = self.create_service(CallMPC, self.mpc_tracking_service_name, self.callback_call_mpc_local)

        # Start timers
        self.gazebo_timer = self.create_timer(0.2, self.gazebo)
        self.save_idle_control_signal_timer = self.create_timer(self.T, self.save_idle_control_signal)

        # Start CPU and memory usage thread
        th.Thread(target=self.cpu_memory_usage).start()

    def callback_sim_status(self, msg):
        """
        This function is the callback from /sim_status and updates the simulation status.

        :param msg: SimStatus message from interfaces_pkg.msg interfaces
        """

        self.sim_status = msg.status

    def callback_odometry(self, msg):
        """
        This function is the callback from the /agv_id/odom topic used to update the AGV state (pose). 
        It calls the call_odometry_publisher function to publish the odometry to the edge and the save_pose_samples 
        function to save the state to file if the data saving is enabled.

        :param msg: Odometry message from nav_msgs.msg interfaces
        """

        # Thread to accelerate the odometry publishing to be subscribed by edge
        th.Thread(target=self.call_odometry_publisher, args=(msg,)).start()

        # Using the function euler_from_quaternion to convert quaternion into Euler angles
        # and then update the pose x0 of the AGV 
        yaw = self.euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)
        self.x0 = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        self.x0_start = True

        # Save data to file
        if self.save_sim_data and self.mpc_run:
            th.Thread(target=self.save_pose_samples, args=(self.x0,)).start()

    def callback_trajectory(self, msg):
        """
        Callback from /agv_id/trajectory topic to update the trajectory of the AGV related to this node.

        :param msg: Trajectory message from interfaces_pkg.msg interfaces
        """

        trajectory = msg.trajectory
        for k, pose in enumerate(trajectory):
            self.trajectory[k, 0] = pose.x
            self.trajectory[k, 1] = pose.y
            self.trajectory[k, 2] = pose.theta
        lock.acquire()
        self.trajectory_update = True
        lock.release()

    def callback_agv_list(self, msg):
        """
        Callback from /agv_list topic to update the list of active AGVs. 
        To reprocess the MPC cost function, it calls the funtion renew_mpc 
        if the AGV priority is changed.

        :param msg: AGVList message from interfaces_pkg.msg interfaces
        """

        for agv in msg.agv_list:
            if agv.agv_id == self.agv_id and agv.priority != self.priority:
                self.priority = agv.priority
                self.renew_mpc()

    def call_odometry_publisher(self, msg):
        """
        This function publishes the odometry to the /agv_id/odom/local topic, which edge subscribes.

        :param msg: Odometry message from nav_msgs.msg interfaces to be published, which edge subscribes.
        """
        self.odometry_publisher.publish(msg)

    def call_velocity_publisher(self, linear, angular):
        """
        This function publishes the control signals to the /agv_id/cmd_vel topic, which Gazebo subscribes.

        :param linear: linear velocity control signal
        :param angular: angular velocity control signal
        """

        msg = Twist()
        msg.linear.x = linear
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = angular
        self.velocity_publisher.publish(msg)

    def callback_call_mpc_local(self, request, response):
        """
        This node is a server of the /call_mpc_local service, and this function is a callback 
        for the service. The function receives a request (goal point or reference) and returns a response.

        :param request: request of type CallMPC from interfaces_pkg.srv
        :param response: response of type CallMPC from interfaces_pkg.srv
        :return: returns the response
        """
        
        if request.action == 'move':
            self.set_xs(request.goal_point.x, request.goal_point.y, request.goal_point.theta)
            self.set_linear_velocity(request.linear_velocity)
            if request.tolerance > 0.0:
                self.tolerance = request.tolerance
            else:
                self.tolerance = 0.3
            if self.first_run:
                self.mpc_timer = self.create_timer(self.T, self.start_mpc)
                self.first_run = False
            else:
                self.mpc_timer.reset()
            self.get_logger().info('Starting MPC for trajectory tracking...')
        if request.action == 'stop':
            if self.mpc_run:
                self.mpc_timer.cancel()
            self.mpc_run = False
            self.call_velocity_publisher(0.0, 0.0)
            self.get_logger().info('Stopping MPC..')
        response.success = True
        return response
    
    def call_close_mpc(self):

        """
        This function calls the /agv_id/mpc/close service of MPC from the edge.
        """

        client = self.create_client(CloseMPC, self.agv_id + '/mpc/close')

        self.get_logger().info('Calling CloseMPC service...')

        request = CloseMPC.Request()
        request.request = 'close'

        future = client.call_async(request)
        future.add_done_callback(partial(self.callback_call_close_mpc, agv_id=self.agv_id))

    def callback_call_close_mpc(self, future, agv_id):
        """
        Callback to treat the response of MPC from the edge when the /agv_id/mpc/close service is called.

        :param future: receives the service response
        :param agv_id: identifier of AGV, who is the client of the service
        """

        try:
            response = future.result()
            if response == True:
                self.get_logger().info(f'CloseMPC service request has been accepted for {agv_id}.')
            if response == False:
                self.get_logger().error(f'CloseMPC service request has not been accepted for{agv_id}.')
        except Exception as e:
            self.get_logger.error('Service call failed %r' % (e,))
    
    def euler_from_quaternion(self, x, y, z, w):
        """
        This function converts a quaternion into Euler angles (roll, pitch, yaw). 
        Roll is rotation around x in radians (counterclockwise), 
        the pitch is rotation around y in radians (counterclockwise), 
        and yaw is rotation around z in radians (counterclockwise).

        :param x: quaternion x
        :param y: quaternion y
        :param z: quaternion z
        :param w: quaternion w
        """
        
        # t0 = +2.0 * (w * x + y * z)
        # t1 = +1.0 - 2.0 * (x * x + y * y)
        # roll_x = math.atan2(t0, t1)
    
        # t2 = +2.0 * (w * y - z * x)
        # t2 = +1.0 if t2 > +1.0 else t2
        # t2 = -1.0 if t2 < -1.0 else t2
        # pitch_y = math.asin(t2)

        # Only the yaw_z is of interest for the AGV navigation
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        # yaw_z in radians
        return yaw_z

    def gazebo(self):
        """
        This function is triggered by a timer and waits for the start of the AGV node in the Gazebo.
        The timer is destroyed when the node starts publishing the odometry.
        """
        nodes = self.get_node_names_and_namespaces()
        topics = self.get_topic_names_and_types()

        if ('differential_drive_controller', '/' + self.agv_id) in nodes:
            if ('/' + self.agv_id + '/odom', ['nav_msgs/msg/Odometry']) in topics and self.x0_start:
                self.gazebo_timer.destroy()
                self.cost_function()

    def renew_mpc(self):
        """
        This function reprocess the MPC cost function.
        """

        if self.mpc_run:
            self.mpc_timer.cancel()
            self.cost_function()
            self.mpc_timer.reset()
        else:
            self.cost_function()

    def cpu_memory_usage(self):
        """
        This function gets the usage from CPU, memory, and swap from the process associated to this node 
        and calls the save_cpu_mem to save data to file if the data saving is enabled.
        """
        while True:
            self.cpu_usage = self.process.cpu_percent(interval=1) / psutil.cpu_count()
            self.mem_usage_percent = self.process.memory_percent()
            self.swap_usage_percent = self.process.memory_percent(memtype="swap")

            # Save data to file
            if self.save_sim_data and self.mpc_run:
                th.Thread(target=self.save_cpu_mem, args=(self.cpu_usage, self.mem_usage_percent, self.swap_usage_percent)).start()

    def save_idle_control_signal(self):
        """
        If the AGV is stopped and simulation data saving is enabled, this function calls the save_control_signal 
        function to save zero value for control signals in those instants.
        """
        if self.save_sim_data and self.sim_status and not self.mpc_run:
            self.save_control_signal(0.0, 0.0)
    
    def save_pose_samples(self, pose: list):
        """
        This function gets the subscribed odometry samples and writes to the trajectory file 
        (the set of pose samples will form the trajectory traveled by the AGV).
        """

        self.trajectory_file.write(f'{pose[0]} {pose[1]} {pose[2]}\n')

    def save_control_signal(self, u1: float, u2: float):
        """
        This function writes to file the control signals.

        :param u1: linear velocity control signal
        :param u2: angular velocity control signal
        """
        self.control_signal_file.write(f'{u1} {u2}\n')

    def save_cpu_mem(self, cpu_usage: float, mem_usage: float, mem_swap: float):
        """
        This function writes to file the usage of CPU, memory, and swap.

        :param cpu_usage: CPU usage
        :param mem_usage: RAM memory usage
        :param mem_swap: Swap memory usage
        """
        if self.save_sim_data:
            self.cpu_mem_file.write(f'{cpu_usage} {mem_usage} {mem_swap}\n')

    def close_files(self):
        """
        This function closes all simulation data files.
        """

        self.get_logger().warn("Closing files...")
        self.trajectory_file.close()
        self.control_signal_file.close()
        self.cpu_mem_file.close()
        self.mpc_time_file.close()
        self.get_logger().warn("Simulation data has been successfully saved.")

        
def main(args=None):
    """
    Node main function.
    """

    rclpy.init(args=args)
    node = MPCLocalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.save_sim_data: 
            node.close_files() # close files if data saving is enabled
        rclpy.shutdown()


if __name__ == "__main__":
    main()