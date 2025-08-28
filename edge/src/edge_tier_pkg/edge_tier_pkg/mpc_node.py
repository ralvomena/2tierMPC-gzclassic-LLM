#!/usr/bin/env python3

import rclpy
import rclpy.qos as qos
import time
import math
import numpy as np
import psutil
import edge_tier_pkg.obstacles as obs
from threading import Thread
from edge_tier_pkg.mpc_core import MPC
from functools import partial
from rclpy.node import Node

# Messages and services interfaces
from interfaces_pkg.msg import AGVList, Trajectory
from interfaces_pkg.srv import RegisterAGV, CallMPC, CloseMPC
from nav_msgs.msg import Odometry

# QoS profile used for the trajectory publisher
qos_profile = qos.QoSProfile(reliability=qos.QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=qos.QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, depth=0)

class MPCEdgeNode(Node, MPC):
    """
    MPC edge node class. This class sets up the ROS environment for MPC. 
    It also saves the simulation data, if enabled.
    """

    def __init__(self):
        """
        MPCEdgeNode constructor.
        """

        # Inherit Node class and pass a default node name (can be changed by a parameter in launch file)
        super().__init__("agv_1_mpc")

        # Inherit MPC class passing the parameters
        super(Node, self).__init__(N=60, T=0.1, Q=np.diag([5, 5, 0.5]), R=np.diag([0.5, 0.5]), ma57_solver=False)

        # Declare and get parameters (all of them can be changed in launch file)
        self.declare_parameter("agv_id", "agv_1")
        self.agv_id = self.get_parameter("agv_id").value
        self.declare_parameter("sim_name", "sim_name")
        self.sim_name = self.get_parameter("sim_name").value
        self.declare_parameter("save_sim_data", False)
        self.save_sim_data = self.get_parameter("save_sim_data").value
        self.declare_parameter("agv_diam", 0.75)
        self.agv_diam = self.get_parameter("agv_diam").value
        self.declare_parameter("d_safe", 1.0)
        self.d_safe = self.get_parameter("d_safe").value
        self.declare_parameter("d_safe_obs", 1.0)
        self.d_safe_obs = self.get_parameter("d_safe_obs").value
        self.declare_parameter("scenario", 1)
        self.scenario = self.get_parameter("scenario").value
        self.declare_parameter("obstacles", 5)
        self.obstacles = self.get_parameter("obstacles").value
        self.declare_parameter("scn_size", [-100, -100, 100, 100]) #-x, -y, +x, +y
        self.scn_size = self.get_parameter("scn_size").value
        self.declare_parameter("limit_n", 0)
        self.limit_n = self.get_parameter("limit_n").value

        # Save the simulation data in this folder
        self.path = '/media/psf/Dropbox/Sim'

        # Open simulation data files
        if self.save_sim_data:
            # self.pkg_id_file = open(self.path +'/pkg_id_' + self.agv_id + '_' + self.sim_name + '_edge.txt', 'w')
            self.cpu_mem_file = open(self.path + '/cpu_mem_' + self.agv_id + '_' + self.sim_name + '_edge.txt', 'w')
            self.mpc_time_file = open(self.path + '/mpc_time_' + self.agv_id + '_' + self.sim_name + '_edge.txt', 'w')
        
        # Terminal node start info
        self.get_logger().info(f"Trajectory tracking MPC node has been started for {self.agv_id.upper()} with {self.scenario}.")
        if self.save_sim_data:
            self.get_logger().info(f"Simulation data will be saved at {self.path} on files terminated with '{self.sim_name}'.")

        # Init some variables
        self.agv_list = []
        self.other_trajectories = {}
        self.sorted_dist = {}
        self.pkg_id_list = []
        self.waiting_info = False
        self.agv_started_publishing = False
        self.first_run = True
        self.process = psutil.Process()
        self.cpu_mem = []
        self.mpc_time_list = []

        # Start publishers and subscribers
        self.trajectory_publisher = self.create_publisher(Trajectory, self.agv_id + '/trajectory', qos_profile)
        self.agv_list_subscriber = self.create_subscription(AGVList, 'agv_list', self.callback_agv_list, 10)
        self.odometry_subscriber = self.create_subscription(Odometry, self.agv_id + '/odom/local', self.callback_odometry,qos.qos_profile_sensor_data)
        
        # Start timers
        self.publish_idle_trajectory_timer = self.create_timer(self.T, self.publish_idle_trajectory)
        self.wait_agv_start_timer = self.create_timer(self.T, self.wait_agv_start)

        # Start timer to trigger the function which calculates the distance from the AGV associated to this node to others AVGs
        if self.limit_n > 0:
            self.calc_dist_other_agvs_timer = self.create_timer(0.01, self.calc_distance_other_agvs)
        
        # Load obstacles and start timer to trigger the function which calculates the distance from the AGV to fixed obstacles
        if self.obstacles > 0:
            obs.load_obstacles(self.scenario)
            self.calc_distance_fixed_obstacles_timer = self.create_timer(0.01, self.calc_distance_fixed_obstacles)
            self.get_logger().info(f"Obstacles from {self.scenario} has been included.")
        
        # Start services
        self.mpc_service_name = self.agv_id + '/mpc/planning'
        self.close_mpc_name = self.agv_id + '/mpc/close'
        self.call_mpc_server_ = self.create_service(CallMPC, self.mpc_service_name, self.callback_call_mpc)
        self.close_mpc = self.create_service(CloseMPC, self.close_mpc_name, self.callback_close_mpc)
        
        # Start CPU and memory usage thread
        Thread(target=self.cpu_memory_usage).start()

    def callback_odometry(self, msg):
        """
        Callback from /agv/odom topic to update AGV state.

        :param msg: Odometry message from nav_msgs.msg interfaces
        """

        yaw = self.euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)

        
        self.x0 = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        self.x0_update = True
        self.agv_started_publishing = True

    def callback_all_trajectories(self, msg):
        """
        Callback from /all_trajectories topic to update the trajectories of others AGVs on self.other_trajectories list.
        The funtion self.update_mpc_parameters is then called to pass these parameters to the MPC, which are used to collision 
        avoidance procedures.

        :param msg: Trajectory message from interfaces_pkg.msg interfaces
        """
        self.other_trajectories.update({msg.agv_id: msg.trajectory})

    def callback_agv_list(self, msg):
        """
        Callback from /agv_list topic to update self.n whenever a new AGV is detected on topic.
        When self.n is updated, renew_mpc() function is called to renew the MPC cost function.

        :param msg: AGVList message from interfaces_pkg.msg interfaces
        """

        for agv in msg.agv_list:
            if agv.agv_id != self.agv_id and agv.agv_id not in self.agv_list:
                OtherAGVs(agv.agv_id, self)
                self.agv_list.append(agv.agv_id)
        
        new_n = len(msg.agv_list)
        if self.limit_n > 0:
            if new_n > 0 and new_n > self.limit_n and self.n != self.limit_n:
                self.n = self.limit_n
                self.renew_mpc()
        else:
            if new_n > 0 and new_n != self.n:
                self.n = new_n
                self.renew_mpc()

    def wait_agv_start(self):
        """
        This functions waits the AGV launching in Gazebo.
        After launching, it starts timer to trigger the function which calculates the distance from the AGV associated 
        to this node to others AVGs, and the timer to trigger the function which updates the MPC parameters.
        """
        if not self.agv_started_publishing:
            if not self.waiting_info:
                self.get_logger().info(f"Waiting for the launch of the {self.agv_id.upper()} local node.")
                self.waiting_info = True
        else:
            self.wait_agv_start_timer.destroy()
            self.call_register_agv_service()
            time.sleep(1)
            self.cost_function()
            time.sleep(1)
            self.calc_dist_other_agvs_timer = self.create_timer(0.01, self.calc_distance_other_agvs)
            time.sleep(1)
            self.update_mpc_parameters_timer = self.create_timer(0.01, self.update_mpc_parameters)

    def call_register_agv_service(self):
        """
        This function calls the /register_agv service on supervisory.
        """

        client = self.create_client(RegisterAGV, "register_agv")
        while not client.wait_for_service(2.0):
            self.get_logger().warn("Waiting for Supervisor start.")
        
        request = RegisterAGV.Request()
        request.agv_id = self.agv_id
        request.priority = 'none'

        future = client.call_async(request)
        future.add_done_callback(partial(self.callback_register_agv, agv_id=self.agv_id))

    def callback_register_agv(self, future, agv_id):
        """
        Callback to treat the response of supervisory when the /register_agv service is called.
        """

        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"{agv_id.upper()} has been succesfully registered on Supervisor.")
            else:
                self.get_logger().warn(f"{agv_id.upper()} is already registered on Supervisor.")
        except Exception as e:
            self.get_logger().error("Service call failed %r" % (e,))

    def callback_call_mpc(self, request, response):
        """
        This node is a server for the CallMPC service, thus, this function is a callback for this service.
        The function receives a request (the goal point) and returns a response.
        :param request: service request.
        :param response: variable to attach the response.
        :return: returns the response.
        """

        response.success = True
        if request.action == 'move':
            self.set_xs(request.goal_point.x, request.goal_point.y, request.goal_point.theta)
            self.set_linear_velocity(request.linear_velocity)
            self.call_mpc_tracking('move', request.goal_point.x, request.goal_point.y, request.goal_point.theta, 
                                   request.linear_velocity, request.tolerance)
            if self.first_run:
                self.timer_ = self.create_timer(self.T, self.start_mpc)
                self.first_run = False
            else:
                self.timer_.reset()
            self.get_logger().info("Starting trajectory planning...")
        if request.action == 'stop':
            if self.mpc_run:
                self.timer_.cancel()
            self.mpc_run = False
            self.call_mpc_tracking('stop', 0.0, 0.0, 0.0, 1.0, 0.3)
        return response

    def call_mpc_tracking(self, action, x, y, theta, linear_velocity, tolerance):
        """
        This function calls the CallMPCLocal service to start the local MPC.
        :param action: action to be done (move, stop...)
        :param x: x target
        :param y: y target
        :param theta: theta target 
        """

        client = self.create_client(CallMPC, self.agv_id + '/mpc/tracking')
        request = CallMPC.Request()
        request.action = action
        request.goal_point.x, request.goal_point.y, request.goal_point.theta = x, y, theta
        request.linear_velocity = linear_velocity
        request.tolerance = tolerance

        future = client.call_async(request)
        future.add_done_callback(partial(self.callback_call_mpc_tracking, agv_id=self.agv_id, action=action, 
                                         x=x, y=y, theta=theta, linear_velocity=linear_velocity))

    def callback_call_mpc_tracking(self, future, agv_id, action, x, y, theta, linear_velocity):
        """
        Callback to treat the response of CallMPCLocal service.
        """

        try:
            response = future.result()
            if response.success:
                if action == 'move' or action == 'stop':
                    self.get_logger().info(f"Trajectory tracking MPC service has received the request.")
            else:
                self.get_logger().warn(f"Trajectory tracking MPC service has not received the request.")
        except Exception as e:
            self.get_logger().error("Service call failed %r" % (e,))

    def callback_close_mpc(self, request, response):
        """
        This node is a server for the CloseMPC service, thus, this function is a callback for this service.
        The function receives a request (close MPC) and returns a response.
        :param request: service request.
        :param response: variable to attach the response.
        :return: returns the response.
        """

        if request.request == 'close' and self.mpc_run:
            self.timer_.cancel()
            self.mpc_run = False
            response.success = True
            self.get_logger().info('The AGV is at reference. Ending planning task...')
            self.get_logger().info(f'Iterations time interval average: {round(self.mpc_time_avg*1000, 2)} ms')
        else:
            response.success = False
        return response
    
    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        # t0 = +2.0 * (w * x + y * z)
        # t1 = +1.0 - 2.0 * (x * x + y * y)
        # roll_x = math.atan2(t0, t1)
    
        # t2 = +2.0 * (w * y - z * x)
        # t2 = +1.0 if t2 > +1.0 else t2
        # t2 = -1.0 if t2 < -1.0 else t2
        # pitch_y = math.asin(t2)
    
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
    
        return yaw_z # in radians

    def renew_mpc(self):
        """
        Renew the cost function when the AGVs quantity is updated.
        """

        if self.mpc_run:
            self.timer_.cancel()
            self.cost_function()
            self.timer_.reset()
        else:
            if self.agv_started_publishing:
                self.cost_function()

    def update_mpc_parameters(self):
        """
        Function to pass the trajectories of others AGVs to the MPC as parameters in order to be used on collision 
        avoidance procedures.
        """

        if self.n > 1:
            i = 0
            if self.limit_n > 0:
                if len(self.sorted_dist.keys()) > 0:
                    keys = list(self.sorted_dist.keys())
                    if len(keys) >= self.limit_n:
                        for k in range(self.limit_n):
                            msg = self.other_trajectories.get(keys[k])
                            for pose in msg:
                                self.col_avoidance_parameters[2*i:2*i+2] = [pose.x, pose.y]
                                i += 1
            else:
                for key in self.other_trajectories.keys():
                    msg = self.other_trajectories.get(key)
                    for pose in msg:
                        self.col_avoidance_parameters[2*i:2*i+2] = [pose.x, pose.y]
                        i += 1

    def cpu_memory_usage(self):
        """
        This function gets the usage from CPU, memory, and swap from the process associated to this node 
        and calls the save_cpu_mem to save data to file if the data saving is enabled.
        """
        while True:
            self.cpu_usage = self.process.cpu_percent(interval=1) / psutil.cpu_count()
            self.mem_usage = self.process.memory_percent()
            self.swap_usage = self.process.memory_percent(memtype="swap")

            if self.save_sim_data and self.mpc_run:
                Thread(target=self.append_cpu_mem, args=(self.cpu_usage, self.mem_usage, self.swap_usage)).start()

    def append_cpu_mem(self, cpu_usage, mem_usage, mem_swap):
        """
        This function writes to file the usage of CPU, memory, and swap.

        :param cpu_usage: CPU usage
        :param mem_usage: RAM memory usage
        :param mem_swap: Swap memory usage
        """
        if self.save_sim_data and self.mpc_run:
            self.cpu_mem_file.write(f'{cpu_usage} {mem_usage} {mem_swap}\n')

    def append_pkg_id(self, pkg_id):
        """
        This function writes to file the received package identifier.

        :param pkg_id: Package identifier
        """

        if self.save_sim_data:
            self.pkg_id_file.write(f'{pkg_id}\n')

    def close_files(self):
        """
        This function closes all simulation data files.
        """
        self.get_logger().warn("Closing files...")
        # self.pkg_id_file.close()
        self.cpu_mem_file.close()
        self.mpc_time_file.close()
        self.get_logger().warn("Simulation data has been successfully saved.")


class OtherAGVs:
    """
    This class is used to deal with the trajectories from the other AGVs.
    """
    def __init__(self, agv_id, node_instance):
        self.agv_id = agv_id
        self.all_trajectories_subscriber = node_instance.create_subscription(Trajectory, self.agv_id + '/trajectory', node_instance.callback_all_trajectories, qos_profile)

def main(args=None):
    rclpy.init(args=args)
    node = MPCEdgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.save_sim_data:
            node.close_files() # close files if data saving is enabled
        rclpy.shutdown()


if __name__ == "__main__":
    main()