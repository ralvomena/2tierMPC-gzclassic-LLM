import rclpy
import os
from rclpy.node import Node
from interfaces_pkg.srv import NetemEdge

class NetemNode(Node):
    """
    Class used to apply NetEm commands.
    It runs a ROS service called NetemEdge, that receives requests from the Netem GUI running at local side.
    """
    def __init__(self):
        super().__init__("edge_netem_node")

        self.timer_enabled = False
        self.filter_enabled = False

        # ip = '192.168.1.1' # Local nodes IP
        # n = 8 # Number of AGVs
        # self.ips = {'agv_'+ str(k):ip for k in range(1, n+1)}

        # If you are running the local nodes in different machines, set up the ip individually for NetEm
        # self.ips = {'agv_1': '192.168.5.118', 'agv_2': '192.168.5.143', 'agv_3': '192.168.5.115', 'agv_4': '192.168.5.185',
        #             'agv_5': '192.168.5.194', 'agv_6': '192.168.5.135', 'agv_7': '192.168.5.173', 'agv_8': '192.168.5.172'}

    # Start services
        self.netem_service_name = 'netem/service/edge'
        self.call_mpc_local = self.create_service(NetemEdge, self.netem_service_name, self.callback_netem)

    def callback_netem(self, request, response):
        agv_id = request.agv_id
        action = request.action
        delay = request.delay
        distribution = request.distribution
        loss = request.loss
        timer = request.timer

        if agv_id == 'all':
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
        else:
            if action == 'enable':
                if delay > 0 and distribution > 0 and loss == 0:
                    self.delay_individual(delay, distribution, agv_id)
                    response.success = True
                elif delay == 0 and loss >= 0:
                    self.loss_individual(loss, agv_id)
                    response.success = True
                elif delay > 0 and distribution > 0 and loss > 0:
                    self.delay_loss_individual(delay, distribution, loss, agv_id)
                    response.success = True
                else:
                    response.success = False
                if timer > 0:
                    self.timer_enabled = True
                    self.netem_timer = self.create_timer(timer, self.delete_individual)
                return response
            if action == 'disable':
                self.delete_individual()
                response.success = True
                return response

        
    def delay(self, delay, distribution):
        os.system(f'sudo tc qdisc add dev eth0 root netem delay {delay}ms {distribution}ms distribution normal')
        self.get_logger().info(f'Netem delay enabled.')

    def loss(self, loss):
       os.system(f'sudo tc qdisc add dev eth0 root netem loss {loss}%')
       self.get_logger().info(f'Netem loss enabled.')

    def delay_loss(self, delay, distribution, loss):
        os.system(f'sudo tc qdisc add dev eth0 root netem delay {delay}ms {distribution}ms distribution normal loss {loss}%')
        self.get_logger().info(f'Netem delay and loss enabled.')

    def delete(self):
        os.system('sudo tc qdisc del dev eth0 root netem')
        if self.timer_enabled:
            self.netem_timer.cancel()
            self.timer_enabled = False
            self.get_logger().info(f'Netem disabled by time.')
        else:
            self.get_logger().info(f'Netem disabled.')

    # def delay_individual(self, delay, distribution, agv_id):
    #     self.get_logger().info(f'{delay},{distribution},{self.ips[agv_id]}')
    #     if not self.filter_enabled:
    #         self.get_logger().info('filter')
    #         self.filter_enabled = True
    #         os.system('sudo tc qdisc add dev eth0 root handle 1: prio')
    #         os.system('sudo tc qdisc add dev eth0 parent 1:3 handle 30: tbf rate 20kbit buffer 1600 limit  3000')
    #         os.system(f'sudo tc qdisc add dev eth0 parent 30:1 handle 31: netem delay {delay}ms {distribution}ms distribution normal')
    #     os.system(f'sudo tc filter add dev eth0 protocol ip parent 1:0 prio 3 u32 match ip dst {self.ips[agv_id]}/32 flowid 1:3')

    # def loss_individual(self, loss, agv_id):
    #     if not self.filter_enabled:
    #         self.filter_enabled = True
    #         os.system('sudo tc qdisc add dev eth0 root handle 1: prio')
    #         os.system('tc qdisc add dev eth0 parent 1:3 handle 30: tbf rate 20kbit buffer 1600 limit  3000')
    #         os.system(f'sudo tc qdisc add dev eth0 parent 30:1 handle 31: netem loss {loss}%')
    #     os.system(f'sudo tc filter add dev eth0 protocol ip parent 1:0 prio 3 u32 match ip dst {self.ips[agv_id]}/32 flowid 1:3')

    # def delay_loss_individual(self, delay, distribution, loss, agv_id):
    #     if not self.filter_enabled:
    #         self.filter_enabled = True
    #         os.system('sudo tc qdisc add dev eth0 root handle 1: prio')
    #         os.system('sudo tc qdisc add dev eth0 parent 1:3 handle 30: tbf rate 20kbit buffer 1600 limit  3000')
    #         os.system(f'sudo tc qdisc add dev eth0 parent 30:1 handle 31: netem delay {delay}ms {distribution}ms distribution normal loss {loss}%')
    #     os.system(f'sudo tc filter add dev eth0 protocol ip parent 1:0 prio 3 u32 match ip dst {self.ips[agv_id]}/32 flowid 1:3')

    # def delete_individual(self):
    #     os.system('sudo tc qdisc del dev eth0 root handle 1: prio')
    #     self.filter_enabled = False
    #     if self.timer_enabled:
    #         self.netem_timer.cancel()
    #         self.timer_enabled = False
    #         self.get_logger().info(f'Netem removido por tempo.')

def main(args=None):
    rclpy.init(args=args)
    node = NetemNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()