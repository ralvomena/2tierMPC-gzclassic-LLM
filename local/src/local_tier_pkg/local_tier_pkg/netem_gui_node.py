import rclpy
import threading
import tkinter as tk
import local_tier_pkg.netem_gui as gui
from tkinter import messagebox
from functools import partial
from rclpy.node import Node
from interfaces_pkg.srv import Netem, NetemEdge

gui_win = None
gui_start = False

class NetemGuiNode(Node):
    """
    Netem GUI node class.
    """
    def __init__(self):
        super().__init__("netem_gui_node")

        # Init some variables
        self.service_names = []
        
        # Start timers
        self.get_service_names_timer = self.create_timer(1, self.get_service_names)

    def get_service_names(self):
        while not gui_start:
            pass
        while not self.get_service_names_and_types():
            pass
        services = self.get_service_names_and_types()
        for service in services:
            if '/netem/service/local' in service[0]:
                if service not in self.service_names:
                    self.service_names.append(service)
                    # agv = service[0][1:6]
                    # gui_win.add_srv_entry(agv)

    def apply_netem_individually(self, agv, action, delay=0, distribution=0, loss=0, timer=0):
        try:

            service_name = agv + '/netem/service/local'
            client = self.create_client(Netem, service_name)
            request = Netem.Request()
            request.action = action
            request.delay = float(delay)
            request.distribution = float(distribution)
            request.loss = float(loss)
            request.timer = float(timer)
            future = client.call_async(request)
            future.add_done_callback(partial(self.callback_apply_netem, agv=agv))

        except ValueError:
            messagebox.showerror('Erro de formato', 'Digite int ou float.')

    def apply_netem_all(self, action, delay=0, distribution=0, loss=0, timer=0):
        try:
            request = Netem.Request()
            request.action = action
            request.delay = float(delay)
            request.distribution = float(distribution)
            request.loss = float(loss)
            request.timer = float(timer)
        except ValueError:
            messagebox.showerror('Erro de formato', 'Digite int ou float.')
            return

        for service in self.service_names:
            client = self.create_client(Netem, service[0])
            future = client.call_async(request)
            future.add_done_callback(partial(self.callback_apply_netem, agv=service[0][1:6]))

        request = NetemEdge.Request()
        request.action = action
        request.agv_id = 'all'
        request.delay = float(delay)
        request.distribution = float(distribution)
        request.loss = float(loss)
        request.timer = float(timer)
        client = self.create_client(NetemEdge, 'netem/service/edge')
        future = client.call_async(request)
        future.add_done_callback(partial(self.callback_apply_netem_edge, agv='all'))
    

    def callback_apply_netem(self, future, agv):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Netem modificado para o {agv.upper()}.')
            else:
                self.get_logger().error(f'Erro no Netem para o {agv.upper()}.')
        except Exception as e:
            self.get_logger().warn().error('Service call failed %r' % (e,))

    def callback_apply_netem_edge(self, future, agv):
        try:
            response = future.result()
            if response.success:
                if agv == 'all':
                    self.get_logger().info(f'Netem modificado para todos os AGVs na borda.')
                else:
                    self.get_logger().info(f'Netem modificado para o {agv.upper()} na borda.')
            else:
                if agv == 'all':
                    self.get_logger().error(f'Erro no Netem para todos os AGVs na borda.')
                else:
                    self.get_logger().error(f'Erro no Netem para o {agv.upper()} na borda.')
        except Exception as e:
            self.get_logger().warn().error('Service call failed %r' % (e,))

    
        

def start_gui(node_instance):
    global gui_win, gui_start
    mainWindow = tk.Tk()
    gui_win = gui.NetemGui(mainWindow, node_instance)
    gui_start = True
    mainWindow.mainloop()
    rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = NetemGuiNode()
    gui = threading.Thread(target=start_gui, args=(node,))
    gui.start()
    rclpy.spin(node)
    rclpy.shutdown()