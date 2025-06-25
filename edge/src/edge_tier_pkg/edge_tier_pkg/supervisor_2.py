#!/usr/bin/env python3
import rclpy
import math
import time
import threading
# import tkinter as tk # Não é mais necessário para a GUI principal
from rclpy.node import Node
from functools import partial
from datetime import datetime

# Importar o módulo da GUI PyQt
import edge_tier_pkg.supervisor_gui_2 as pyqt_gui_module # Nome do arquivo da GUI PyQt
from PyQt5.QtWidgets import QApplication # Ou PyQt6.QtWidgets
import sys # Necessário para QApplication e sys.exit

# Import scenarios classes
from edge_tier_pkg.scenario1 import Scenario1
# from edge_tier_pkg.scenario2 import Scenario2
from edge_tier_pkg.scenario2_llm import Scenario2 # Supondo que esta é a desejada
from edge_tier_pkg.scenario3 import Scenario3
from edge_tier_pkg.scenario4 import Scenario4

# Messages and services interfaces
from interfaces_pkg.msg import AGVMsg, AGVList, Trajectory, SimStatus
from interfaces_pkg.srv import RegisterAGV, CallMPC
from nav_msgs.msg import Odometry

# Variáveis globais para a GUI PyQt
pyqt_app_instance = None # Para a instância QApplication
pyqt_gui_window_instance = None # Para a instância da janela principal da GUI
pyqt_gui_has_started = False # Flag

class SupervisorNode(Node, Scenario2): #Herda a classe do cenário que você quer simular
    """
    Supervisor node class.
    """
    def __init__(self):
        super().__init__("supervisor_node")
        super(Node, self).__init__() # Esta linha pode ser redundante se Scenario2 já lida com a herança de Node
                                      # Ou se Scenario2 não herda de Node, então a chamada super() padrão é suficiente.
                                      # A chamada super() sem argumentos é geralmente preferida em Python 3.

        # Terminal node start info
        self.get_logger().info("Supervisor started.")

        # Init some variables
        self.agv_list = []
        self.agv_instances = []
        self.all_trajectories = []
        self.run = False # Para SimStatus

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
        msg = SimStatus()
        msg.status = self.run
        self.sim_status_publisher.publish(msg)

    def publish_agv_list(self):
        msg_agv_list = AGVList()
        agv_list_msgs = [] # Renomeado para evitar conflito com self.agv_list
        for agv in self.agv_instances:
            msg_agv = AGVMsg()
            msg_agv.agv_id = agv.agv_id
            msg_agv.priority = agv.priority
            agv_list_msgs.append(msg_agv)
        msg_agv_list.agv_list = agv_list_msgs
        self.avg_list_publisher.publish(msg_agv_list)

    def set_agv_priority(self, agv_id, priority):
        for agv in self.agv_instances:
            if agv.agv_id == agv_id:
                agv.priority = priority
                self.get_logger().info(f"Prioridade do {agv.agv_id.upper()} ajustada para {priority}.")
                # A atualização na GUI PyQt será feita pelo método update_gui ou diretamente na classe da GUI
                # se a prioridade for alterada por lá.
                return None

    def callback_trajectory(self, msg):
        # AGV.get_instances() precisa ser gerenciado com cuidado se AGV é instanciado aqui
        # Esta parte parece depender de uma implementação específica de AGV.get_instances()
        # Por agora, vamos assumir que self.agv_instances é a fonte da verdade para instâncias AGV.
        for agv_instance in self.agv_instances:
            if agv_instance.agv_id == msg.agv_id:
                agv_instance.trajectory = msg.trajectory
                break

    def callback_register_agv(self, request, response):
        if request.agv_id in self.agv_list:
            self.get_logger().warn(f"{request.agv_id.upper()} is already registered.")
            response.success = False
        else:
            # Passa 'self' (SupervisorNode instance) para o AGV
            agv = AGV(request.agv_id, request.priority, self)
            self.get_logger().info(f"{request.agv_id.upper()} has been successfully registered on Supervisor.")
            self.agv_list.append(request.agv_id)
            self.agv_list.sort() # Mantém a lista de IDs ordenada
            self.agv_instances.append(agv) # Adiciona a instância AGV
            self.agv_instances.sort(key=lambda x: x.agv_id) # Mantém a lista de instâncias ordenada por ID

            if pyqt_gui_has_started and pyqt_gui_window_instance:
                # Pose e velocity iniciais podem ser [0.0,0.0,0.0] ou buscar de algum estado inicial se disponível
                pyqt_gui_window_instance.insert_agv_to_tree(agv.agv_id, [0.0, 0.0, 0.0], [0.0, 0.0], agv.priority, 'On', 'Off')
                log_msg = f"[{datetime.now().strftime('%d-%m-%Y %H:%M:%S.%f')}] {request.agv_id.upper()} has been successfully registered on Supervisor.\n"
                pyqt_gui_window_instance._log_to_info_panel(log_msg) # Usando o helper da GUI PyQt

            response.success = True
        return response

    def update_node_status(self):
        nodes = self.get_node_names_and_namespaces() # Retorna uma lista de tuplas (name, namespace)
        node_names_only = [name for name, ns in nodes] # Extrai apenas os nomes

        for agv in self.agv_instances:
            # Verifica se o nome completo do nó (sem namespace inicial /) existe
            agv.mpc_edge_status = (agv.agv_id + '_mpc') in node_names_only
            agv.mpc_local_status = (agv.agv_id + '_mpc_local_tracking') in node_names_only


    def update_gui(self):
        if pyqt_gui_has_started and pyqt_gui_window_instance:
            for agv in self.agv_instances:
                mpc_edge_status_str = 'On' if agv.mpc_edge_status else 'Off'
                mpc_local_status_str = 'On' if agv.mpc_local_status else 'Off'

                # Assegurar que temos pose e velocity antes de tentar acessar
                if agv.pose and len(agv.pose) == 3 and agv.velocity and len(agv.velocity) == 2:
                    x = str(round(agv.pose[0], 2))
                    y = str(round(agv.pose[1], 2))
                    # Pose[2] (yaw) está em radianos, converter para graus para display
                    theta = str(round(math.degrees(agv.pose[2]), 2))
                    linear = str(round(agv.velocity[0], 2))
                    angular = str(round(agv.velocity[1], 2))

                    if agv.priority == 'high':
                        priority_str = 'High'
                    elif agv.priority == 'medium':
                        priority_str = 'Medium'
                    else: # low ou qualquer outro valor
                        priority_str = 'Low'

                    # Atualizar o item na QTreeWidget
                    agv_id_upper = agv.agv_id.upper()
                    item_to_update = pyqt_gui_window_instance.tree_inserts.get(agv_id_upper)
                    if item_to_update:
                        item_to_update.setText(0, agv_id_upper)
                        item_to_update.setText(1, f'[{x}, {y}, {theta}]')
                        item_to_update.setText(2, f'[{linear}, {angular}]')
                        item_to_update.setText(3, priority_str)
                        item_to_update.setText(4, mpc_edge_status_str)
                        item_to_update.setText(5, mpc_local_status_str)
                # else:
                #    self.get_logger().debug(f"AGV {agv.agv_id} sem pose/velocity completa para atualizar GUI.")


class AGV:
    # __instances = [] # Removido para evitar problemas com múltiplas instanciações do SupervisorNode ou recargas
    # __trajectories = {} # Idem

    def __init__(self, agv_id, priority, supervisor_node_instance): # Renomeado node_instance para clareza
        self.__agv_id = agv_id
        self.__trajectory = []
        self.__pose = []       # Formato: [x, y, yaw_radians]
        self.__velocity = []   # Formato: [linear_x, angular_z]
        self.__priority = priority
        self.__mpc_service_name = f"/{self.__agv_id}/mpc/planning" # Adicionado '/' no início para ser um nome absoluto
        self.__mpc_edge_status = False
        self.__mpc_local_status = False
        self.__status = 'parked' # Não usado no setter `tatus`
        self.supervisor_node = supervisor_node_instance # Referência ao nó Supervisor
        
        # Subscribers devem ser criados pelo nó ROS (SupervisorNode)
        self.odometry_subscriber = self.supervisor_node.create_subscription(
            Odometry,
            f"/{self.__agv_id}/odom", # Adicionado '/' no início
            self.callback_odometry,
            10
        )
        # AGV.add_instance(self) # Removido, gerenciamento de instâncias agora é em SupervisorNode

    @property
    def agv_id(self): return self.__agv_id
    @property
    def mpc_service_name(self): return self.__mpc_service_name
    @property
    def trajectory(self): return self.__trajectory
    @trajectory.setter
    def trajectory(self, trajectory): self.__trajectory = trajectory
    @property
    def pose(self): return self.__pose
    @pose.setter
    def pose(self, pose): self.__pose = pose
    @property
    def velocity(self): return self.__velocity
    @velocity.setter
    def velocity(self, velocity): self.__velocity = velocity
    @property
    def priority(self): return self.__priority
    @priority.setter
    def priority(self, priority): self.__priority = priority
    @property
    def mpc_edge_status(self): return self.__mpc_edge_status
    @mpc_edge_status.setter
    def mpc_edge_status(self, status): self.__mpc_edge_status = status
    @property
    def mpc_local_status(self): return self.__mpc_local_status
    @mpc_local_status.setter
    def mpc_local_status(self, status): self.__mpc_local_status = status
    @property
    def status(self): return self.__status
    @status.setter
    def status(self, status): self.__status = status # Corrigido de `tatus` para `status`

    def callback_odometry(self, msg):
        yaw = self.euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        self.__pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        self.__velocity = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

    def euler_from_quaternion(self, x, y, z, w):
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return yaw_z

    def call_mpc(self, action, x, y, theta, linear_velocity, tolerance=None): # theta em graus
        if self.__mpc_edge_status and self.__mpc_local_status:
            client = self.supervisor_node.create_client(CallMPC, self.__mpc_service_name)
            # Esperar pelo serviço pode ser uma boa prática em alguns casos
            # if not client.wait_for_service(timeout_sec=1.0):
            #     self.supervisor_node.get_logger().error(f"Serviço MPC {self.__mpc_service_name} não disponível.")
            #     # ... logar na GUI também
            #     return

            request = CallMPC.Request()
            request.action = action
            request.goal_point.x = float(x)
            request.goal_point.y = float(y)
            request.goal_point.theta = math.radians(float(theta)) # Convertido para radianos
            request.linear_velocity = float(linear_velocity)

            if tolerance:
                request.tolerance = float(tolerance)

            future = client.call_async(request)
            future.add_done_callback(partial(self.callback_call_mpc, agv_id=self.__agv_id, action=action,
                                             x=x, y=y, theta=theta, linear_velocity=linear_velocity))
        else:
            log_msg_prefix = f"[{datetime.now().strftime('%d-%m-%Y %H:%M:%S.%f')}]"
            error_detail = f"MPC edge ou local do {self.__agv_id.upper()} está desligado."
            if action == 'move':
                self.supervisor_node.get_logger().error(f"Erro na chamada do serviço: {error_detail}")
                if pyqt_gui_has_started and pyqt_gui_window_instance:
                    pyqt_gui_window_instance._log_to_info_panel(f"{log_msg_prefix} Erro na chamada do serviço: {error_detail}\n")
            elif action == 'stop': # 'stop' também requer MPCs online
                self.supervisor_node.get_logger().error(f"Erro na solicitação de parada: {error_detail}")
                if pyqt_gui_has_started and pyqt_gui_window_instance:
                    pyqt_gui_window_instance._log_to_info_panel(f"{log_msg_prefix} Erro na solicitação de parada: {error_detail}\n")


    def callback_call_mpc(self, future, agv_id, action, x, y, theta, linear_velocity):
        log_msg_prefix = f"[{datetime.now().strftime('%d-%m-%Y %H:%M:%S.%f')}]"
        try:
            response = future.result()
            if response.success:
                if action == 'move':
                    log_info = f"Calling MPC service for {agv_id.upper()} with x:{x}, y:{y}, theta:{theta}, v:{linear_velocity}.\n"
                    self.supervisor_node.get_logger().info(log_info)
                    if pyqt_gui_has_started and pyqt_gui_window_instance:
                        pyqt_gui_window_instance._log_to_info_panel(f"{log_msg_prefix} {log_info}")
                elif action == 'stop': # 'stop' também deve ser logado se bem sucedido
                    log_info = f"Parando o {agv_id.upper()}...\n"
                    self.supervisor_node.get_logger().info(log_info)
                    if pyqt_gui_has_started and pyqt_gui_window_instance:
                         pyqt_gui_window_instance._log_to_info_panel(f"{log_msg_prefix} {log_info}")
            else:
                warn_msg = "Service call denied by MPC."
                self.supervisor_node.get_logger().warn(warn_msg)
                if pyqt_gui_has_started and pyqt_gui_window_instance:
                    pyqt_gui_window_instance._log_to_info_panel(f"{log_msg_prefix} {warn_msg}\n")
        except Exception as e:
            error_msg = f"Service call failed for {agv_id.upper()}: {e!r}"
            self.supervisor_node.get_logger().error(error_msg)
            if pyqt_gui_has_started and pyqt_gui_window_instance:
                pyqt_gui_window_instance._log_to_info_panel(f"{log_msg_prefix} {error_msg}\n")

def start_gui_pyqt(node_instance_arg): # Renomeado argumento para evitar conflito com 'node' em main
    """
    Inicializa e executa a interface gráfica PyQt.
    O rclpy.shutdown() é chamado após a GUI ser fechada.
    """
    global pyqt_app_instance, pyqt_gui_window_instance, pyqt_gui_has_started

    pyqt_app_instance = QApplication.instance()
    if pyqt_app_instance is None:
        pyqt_app_instance = QApplication(sys.argv)
    
    pyqt_gui_window_instance = pyqt_gui_module.SupervisorGuiPyQt(node_instance_arg)
    pyqt_gui_window_instance.show()
    pyqt_gui_has_started = True
    
    exit_code = pyqt_app_instance.exec_()
    
    node_instance_arg.get_logger().info("Fechando a GUI PyQt e preparando para desligar o rclpy...")
    # A flag run do nó supervisor pode ser usada para sinalizar outras partes para parar
    node_instance_arg.run = False # Exemplo de como sinalizar o fim da simulação
    # Não chamar rclpy.shutdown() aqui diretamente se o spin principal ainda precisa dele.
    # O shutdown será tratado no bloco finally do `main` ou quando o `rclpy.spin` terminar.
    # A GUI ter sido fechada não significa necessariamente que o nó ROS inteiro deva parar imediatamente,
    # pode haver limpeza a ser feita pelo rclpy.spin.
    # No entanto, se o comportamento original era fechar tudo, pode-se manter rclpy.shutdown() aqui
    # mas é mais robusto deixar o `main` lidar com o shutdown final do rclpy.
    
    # Se o rclpy.shutdown() for chamado aqui, o sys.exit(exit_code) é apropriado.
    # Se não, apenas retorne ou deixe a thread terminar.
    # Para manter o comportamento o mais próximo do original:
    if rclpy.ok():
        rclpy.shutdown() # Desliga o rclpy quando a GUI é fechada.
    sys.exit(exit_code)


def main(args=None):
    rclpy.init(args=args)
    supervisor_node = SupervisorNode()

    # Função para rodar rclpy.spin() em uma thread separada
    def spin_ros_node_in_thread(node_to_spin):
        node_to_spin.get_logger().info("Thread de spin do ROS iniciada.")
        try:
            rclpy.spin(node_to_spin)
        except (KeyboardInterrupt, SystemExit):
            node_to_spin.get_logger().info("Thread de spin do ROS interrompida.")
        except Exception as e:
            node_to_spin.get_logger().error(f"Erro na thread de spin do ROS: {e}")
        finally:
            # Quando rclpy.spin retorna (por exemplo, nó destruído ou rclpy.shutdown chamado),
            # esta thread será finalizada.
            node_to_spin.get_logger().info("Thread de spin do ROS finalizada.")

    # Inicia rclpy.spin na thread secundária
    ros_spin_thread = threading.Thread(target=spin_ros_node_in_thread, args=(supervisor_node,))
    ros_spin_thread.daemon = True  # Permite que a thread saia quando a thread principal (GUI) sair
    ros_spin_thread.start()

    # Chama a função da GUI na thread principal.
    # Esta função irá bloquear até que a GUI seja fechada.
    # A função start_gui_pyqt (da resposta anterior) já lida com a criação de QApplication,
    # .show(), .exec_(), e também chama rclpy.shutdown() e sys.exit().
    try:
        start_gui_pyqt(supervisor_node) # Esta chamada bloqueia e eventualmente chama sys.exit()
    except Exception as e:
        supervisor_node.get_logger().error(f"Erro crítico ao iniciar ou executar a GUI: {e}")
        # Se a GUI falhar catastroficamente, precisamos parar o ROS manualmente.
        if rclpy.ok():
            if hasattr(supervisor_node, 'destroy_node') and callable(getattr(supervisor_node, 'destroy_node')):
                 supervisor_node.destroy_node()
            rclpy.shutdown()