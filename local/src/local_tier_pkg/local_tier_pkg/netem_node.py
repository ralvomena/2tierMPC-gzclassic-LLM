import rclpy
import os
from rclpy.node import Node
from interfaces_pkg.srv import Netem

class NetemNode(Node):
    """
    Class used to apply NetEm commands.
    It runs a ROS service also called Netem, that receives requests from the Netem GUI.
    """
    def __init__(self):
        super().__init__("agv_1_netem_node")
        self.declare_parameter("agv_id", "agv_1")
        self.agv_id = self.get_parameter("agv_id").value
        # self.declare_parameter("edge", "edge_1")
        # self.edge = self.get_parameter("edge").value
        # if self.edge == "edge_1":
        #     self.edge_ip = '192.168.5.120'
        # elif self.edge == "edge_2":
        self.edge_ip = '192.168.5.119'
        self.timer_enabled = False

    # Start services
        self.netem_service_name = self.agv_id + '/netem/service/local'
        self.call_mpc_local = self.create_service(Netem, self.netem_service_name, self.callback_netem)

    def callback_netem(self, request, response):
        action = request.action
        delay = request.delay
        distribution = request.distribution
        loss = request.loss
        timer = request.timer

        if action == 'enable':
            if delay > 0 and distribution > 0 and loss == 0:
                self.delay(delay, distribution)
                response.success = True
            elif delay == 0 and loss >= 0:
                self.loss(loss)
                response.success = True
            elif delay > 0 and distribution > 0 and loss > 0:
                self.delay_loss(delay, distribution, loss)
                response.success = True
            else:
                response.success = False
            if timer > 0:
                self.timer_enabled = True
                self.netem_timer = self.create_timer(timer, self.delete)
            return response
        if action == 'disable':
            self.delete()
            response.success = True
            return response

        
    def delay(self, delay, distribution):
        os.system('sudo tc qdisc add dev eth0 root handle 1: prio')
        os.system(f'sudo tc filter add dev eth0 parent 1:0 protocol ip prio 1 u32 match ip dst {self.edge_ip} flowid 2:1')
        os.system(f'sudo tc qdisc add dev eth0 parent 1:1 handle 2: netem delay {delay}ms {distribution}ms distribution normal')

    def loss(self, loss):
        os.system('sudo tc qdisc add dev eth0 root handle 1: prio')
        os.system(f'sudo tc filter add dev eth0 parent 1:0 protocol ip prio 1 u32 match ip dst {self.edge_ip} flowid 2:1')
        os.system(f'sudo tc qdisc add dev eth0 parent 1:1 handle 2: netem loss {loss}%')


    def delay_loss(self, delay, distribution, loss):
        os.system('sudo tc qdisc add dev eth0 root handle 1: prio')
        os.system(f'sudo tc filter add dev eth0 parent 1:0 protocol ip prio 1 u32 match ip dst {self.edge_ip} flowid 2:1')
        os.system(f'sudo tc qdisc add dev eth0 parent 1:1 handle 2: netem delay {delay}ms {distribution}ms distribution normal loss {loss}%')

    def delete(self):
        os.system('sudo tc qdisc del dev eth0 root handle 1: prio')

        if self.timer_enabled:
            self.netem_timer.cancel()
            self.timer_enabled = False
            self.get_logger().info(f'Netem disabled by time for {self.agv_id}.')
        else:
            self.get_logger().info(f'Netem disabled.')


def main(args=None):
    rclpy.init(args=args)
    node = NetemNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()