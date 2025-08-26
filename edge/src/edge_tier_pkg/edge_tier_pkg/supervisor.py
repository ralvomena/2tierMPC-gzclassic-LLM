#!/usr/bin/env python3
import rclpy
import math
import time
import threading
import tkinter as tk
from rclpy.node import Node
from functools import partial
from datetime import datetime

# Import GUI
import edge_tier_pkg.supervisor_gui as gui

# Import scenarios classes
from edge_tier_pkg.scenario1 import Scenario1
from edge_tier_pkg.scenario2_llm import Scenario2
from edge_tier_pkg.scenario3 import Scenario3
from edge_tier_pkg.scenario4 import Scenario4

# Messages and services interfaces
from interfaces_pkg.msg import AGVMsg, AGVList, Trajectory, SimStatus
from interfaces_pkg.srv import RegisterAGV, CallMPC
from nav_msgs.msg import Odometry

gui_win = None
gui_start = False

class SupervisorNode(Node, Scenario2): #Inherit the scenario class you want to simulate
    """
    Supervisor node class. The Supervisor detects all the AGVs with local nodes on and register them.
    It has a GUI, that will provide informations of simulations and from where you can call the MPC service individually for each AGV.
    If the scenario is started from GUI, the Supervisor calls the MPC services to move AGVs and attend tasks, according to the loaded scenario.

    """
    def __init__(self):

        super().__init__("supervisor_node")
        super(Node, self).__init__()

        # Terminal node start info
        self.get_logger().info("Supervisor started.")

        # Init some variables
        self.agv_list = []
        self.agv_instances = []
        self.all_trajectories = []
        self.run = False

        # Start publishers and subscribers
        self.avg_list_publisher = self.create_publisher(AGVList, 'agv_list', 10)
        self.sim_status_publisher = self.create_publisher(SimStatus, 'sim_status', 10)

        # Start timers
        self.agv_list_publisher_timer = self.create_timer(1.0, self.publish_agv_list)
        self.update_node_status_timer = self.create_timer(0.1, self.update_node_status)
        self.update_gui_timer = self.create_timer(1.0, self.update_gui)
        self.sim_status_publisher_timer = self.create_timer(0.01, self.publish_sim_status)

        # Start services
        self.register_agv_server = self.create_service(RegisterAGV, "register_agv", self.callback_register_agv)

    def publish_sim_status(self):
        """
        Function to publish simulation status.
        """

        msg = SimStatus()
        msg.status = self.run
        self.sim_status_publisher.publish(msg)

    def publish_agv_list(self):
        """
        Called periodically by self.agv_list_publisher_timer to publish the AGVs list on /agv_list topic.
        """

        msg_agv_list = AGVList()
        agv_list = []
        for agv in self.agv_instances:
            msg_agv = AGVMsg()
            msg_agv.agv_id = agv.agv_id
            msg_agv.priority = agv.priority
            agv_list.append(msg_agv)
        msg_agv_list.agv_list = agv_list
        self.avg_list_publisher.publish(msg_agv_list)

    
    def set_agv_priority(self, agv_id, priority):
        """
        Function called by GUI to set agv priority.
        """

        for agv in self.agv_instances:
            if agv.agv_id == agv_id:
                agv.priority = priority
                self.get_logger().info(f"Prioridade do {agv.agv_id.upper()} ajustada para {priority}.")
                return None


    def callback_trajectory(self, msg):

        """
        Callback from /trajectories to update the trajectories on all AGVs instances.
        """
        
        agv_instances = AGV.get_instances()
        if agv_instances:
            for agv in agv_instances:
                agv_id = agv.agv_id
                if agv_id == msg.agv_id:
                    agv.trajectory = msg.trajectory

    def callback_register_agv(self, request, response):
        """
        This node is a server for the RegisterAGV service, thus, this function is a callback for this service.
        The function receives a request (the agv identifier) and returns a response.
        :param request: service request.
        :param response: variable to attach the response.
        :return: returns the response.
        """

        if request.agv_id in self.agv_list:
            self.get_logger().warn(f"{request.agv_id.upper()} is already registered.")
            response.success = False
        else:
            agv = AGV(request.agv_id, request.priority, self)
            self.get_logger().info(f"{request.agv_id.upper()} has been successfully registered on Supervisor.")
            self.agv_list.append(request.agv_id)
            self.agv_list.sort()
            self.agv_instances.append(agv)
            gui_win.insert_agv_to_tree(agv.agv_id, [0.0, 0.0, 0.0], [0.0, 0.0], ' ', 'On', 'Off')
            response.success = True
            # gui_win.add_mpc_entry(request.agv_id)
            gui_win._log_info(f"{request.agv_id.upper()} has been successfully registered on Supervisor.")
        return response

    def update_node_status(self):

        """
        Function used to update status of mpc nodes from edge and local.
        """

        nodes = self.get_node_names()
        for agv in self.agv_instances:
            if agv.agv_id + '_mpc' in nodes:
                agv.mpc_edge_status = True
            else:
                agv.mpc_edge_status = False
            if agv.agv_id + '_mpc_local_tracking' in nodes:
                agv.mpc_local_status = True
            else:
                agv.mpc_local_status = False

    def update_gui(self):
        """
        Function used to update the GUI table (tree).
        """

        if gui_start:
            for agv in self.agv_instances:
                if agv.mpc_edge_status:
                    mpc_edge_status = 'On'
                else:
                    mpc_edge_status = 'Off'
                if agv.mpc_local_status:
                    mpc_local_status = 'On'
                else:
                    mpc_local_status = 'Off'
                if len(gui_win.tree_inserts) > 0:
                    if len(agv.pose) > 0:
                        x = str(round(agv.pose[0], 2))
                        y = str(round(agv.pose[1], 2))
                        linear = str(round(agv.velocity[0], 2))
                        angular = str(round(agv.velocity[1], 2))
                        theta = str(round(math.degrees(agv.pose[2]), 2))
                        if isinstance(agv.target_velocity, float):
                            target_velocity = str(round(agv.target_velocity, 2))
                        else:
                            target_velocity = 'Not set'
                        gui_win.agv_tree.item(gui_win.tree_inserts[agv.agv_id], values=(agv.agv_id.upper(), f'[{x}, {y}, {theta}]', f'[{linear}, {angular}]', target_velocity, mpc_edge_status, mpc_local_status))

    def show_llm_response_gui(self, response):
        gui_win.llm_response_text.insert(tk.END, response)
                         
class AGV:
    """
    Class to manage the AGVs connected to supervisor.
    """

    __instances = []
    __trajectories = {}

    @classmethod
    def add_instance(cls, instance):
        cls.__instances.append(instance)

    @classmethod
    def get_instances(cls):
        return cls.__instances

    @classmethod
    def get_trajectories(cls):
        return cls.__trajectories

    def __init__(self, agv_id, priority, node_instance):
        self.__agv_id = agv_id
        self.__trajectory = []
        self.__pose = []
        self.__target_velocity = []
        self.__velocity = []
        self.__priority = priority
        self.__mpc_service_name = self.__agv_id + '/mpc/planning'
        self.__mpc_edge_status = False
        self.__mpc_local_status = False
        self.__status = 'parked'
        self.node_instance = node_instance
        self.odometry_subscriber = self.node_instance.create_subscription(Odometry, self.__agv_id + '/odom', self.callback_odometry, 10)
        AGV.add_instance(self)


    @property
    def agv_id(self):
        return self.__agv_id

    @property
    def mpc_service_name(self):
        return self.__mpc_service_name

    @property
    def trajectory(self):
        return self.__trajectory

    @trajectory.setter
    def trajectory(self, trajectory):
        self.__trajectory = trajectory
        msg = Trajectory()
        msg.agv_id = self.__agv_id
        msg.trajectory = self.__trajectory
        AGV.__trajectories.update({self.__agv_id: msg})

    @property
    def pose(self):
        return self.__pose

    @pose.setter
    def pose(self, pose):
        self.__pose = pose

    @property
    def target_velocity(self):
        return self.__target_velocity

    @target_velocity.setter
    def target_velocity(self, target_velocity):
        self.__target_velocity = target_velocity

    @property
    def velocity(self):
        return self.__velocity

    @velocity.setter
    def velocity(self, velocity):
        self.__velocity = velocity

    @property
    def priority(self):
        return self.__priority

    @priority.setter
    def priority(self, priority):
        self.__priority = priority

    @property
    def mpc_edge_status(self):
        return self.__mpc_edge_status

    @mpc_edge_status.setter
    def mpc_edge_status(self, status):
        self.__mpc_edge_status = status

    @property
    def mpc_local_status(self):
        return self.__mpc_local_status

    @mpc_local_status.setter
    def mpc_local_status(self, status):
        self.__mpc_local_status = status

    @property
    def status(self):
        return self.__status

    @status.setter
    def tatus(self, status):
        self.__status = status

    def callback_odometry(self, msg):
        """
        Callback from /agv/odom topic to update AGV state.
        """

        yaw = self.euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)

        
        self.__pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        self.__velocity = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

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

    def call_mpc(self, action, x, y, theta, linear_velocity, tolerance=None):
        """
        This function calls the mpc service on mpc edge node.
        """

        if self.__mpc_edge_status and self.__mpc_local_status:
            client = self.node_instance.create_client(CallMPC, self.__mpc_service_name)
            request = CallMPC.Request()

            request.action = action
            request.goal_point.x = x
            request.goal_point.y = y
            request.goal_point.theta = math.radians(theta)
            request.linear_velocity = linear_velocity

            if tolerance:
                request.tolerance = tolerance

            future = client.call_async(request)
            future.add_done_callback(partial(self.callback_call_mpc, agv_id=self.__agv_id, action=action, 
                                             x=x, y=y, theta=theta, linear_velocity=linear_velocity))
        else:
            if action == 'move':
                self.node_instance.get_logger().error(f"MPC edge or local from {self.__agv_id.upper()} is offline.")
                gui_win._log_info(f"Erro na chamada do serviço: o MPC da borda ou local do {self.__agv_id.upper()} está desligado.")
            if action == 'stop':
                self.node_instance.get_logger().error(f"MPC from {self.__agv_id.upper()} is offline.")
                gui_win._log_info(f"Erro na solicitação de parada: o MPC da borda ou local do {self.__agv_id.upper()} está desligado.")
    
    def callback_call_mpc(self, future, agv_id, action, x, y, theta, linear_velocity):
        """
        Callback to treat the response of mpc edge node when the mpc service is called.
        """

        try:
            response = future.result()
            if response.success:
                if action == 'move':
                    self.node_instance.get_logger().info(f"Calling MPC service for {agv_id.upper()} with x:{x}, y:{y}, theta:{theta}, v:{linear_velocity}.\n")
                    gui_win._log_info(f"Calling MPC service for {agv_id.upper()} with x:{x}, y:{y}, theta:{theta}, v:{linear_velocity}.")
                if action == 'stop':
                    self.node_instance.get_logger().info(f"Parando o {agv_id.upper()}...\n")
            else:
                self.node_instance.get_logger().warn("Service call denied.")
        except Exception as e:
            self.node_instance.get_logger().error("Service call failed %r" % (e,))

def start_gui(node_instance):
    global gui_win, gui_start
    mainWindow = tk.Tk()
    gui_win = gui.SupervisorGui(mainWindow, node_instance)
    gui_start = True
    mainWindow.mainloop()
    rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = SupervisorNode()
    time.sleep(2)
    gui = threading.Thread(target=start_gui, args=(node,))
    gui.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        rclpy.shutdown()
