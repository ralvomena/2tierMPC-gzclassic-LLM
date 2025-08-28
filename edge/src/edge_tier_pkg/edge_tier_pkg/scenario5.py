import math
import time
import threading
import numpy as np
from scipy.interpolate import CubicSpline
import requests
import json

# Importações do ROS 2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PathGenerator:
    """
    Gera uma trajetória suave (usando spline cúbica) para o AGV seguir,
    incluindo a orientação (theta) em cada ponto.
    """
    def __init__(self, obstacles: list, lateral_offset: float, start_y: float, end_y: float):
        self.points = []
        start_pose = [0.0, start_y, math.pi / 2]
        self._generate_s_curve_path(obstacles, lateral_offset, start_pose, start_y, end_y)

    def _generate_s_curve_path(self, obstacles, lateral_offset, start_pose, start_y, end_y):
        print(f"[PathGenerator] A gerar trajetória de y={start_y:.1f} para y={end_y:.1f}...")
        
        control_points_x = [start_pose[0]]
        control_points_y = [start_pose[1]]
        
        current_offset = -lateral_offset if start_y < end_y else lateral_offset
        sorted_obstacles = sorted(obstacles, key=lambda p: p[1], reverse=(start_y > end_y))

        for obs_x, obs_y in sorted_obstacles:
            control_points_x.append(obs_x + current_offset)
            control_points_y.append(obs_y)
            current_offset *= -1

        final_x = start_pose[0]
        control_points_x.append(final_x)
        control_points_y.append(end_y)

        combined_points = sorted(zip(control_points_y, control_points_x))
        sorted_y, sorted_x = zip(*combined_points)

        cs = CubicSpline(sorted_y, sorted_x)
        cs_derivative = cs.derivative(1)

        num_points = int(abs(end_y - start_y) * 20)
        y_path = np.linspace(start_y, end_y, num_points)
        x_path = cs(y_path)

        dxd_dy = cs_derivative(y_path)
        angles = np.arctan2(np.ones_like(dxd_dy), dxd_dy)

        self.points = list(zip(x_path, y_path, angles))
        print(f"[PathGenerator] Trajetória com {len(self.points)} pontos gerada.")


class Scenario5:
    """
    Lógica do cenário de controlo. Projetada para ser herdada por uma classe Node do ROS 2,
    como 'SupervisorNode(Node, Scenario5)'.
    """
    def __init__(self):
        # Configurações do cenário
        self.obstacles = [(0, -10), (0, -5), (0, 0), (0, 5), (0, 10)]
        self.lookahead_distance = 2.0
        
        # Trajetórias e pontos de estacionamento serão definidos dinamicamente em start_scenario
        self.forward_path = []
        self.backward_path = []
        self.end_parking_spot = None
        self.start_parking_spot = None

        self.agv_state = 'IDLE'
        self.planned_path = []
        self.executed_path = []
        self.has_converged = False

        self.ollama_ip = "192.168.5.150"
        self.ollama_port = 11434 
        self.llm_model = "phi3"
        self.ollama_url = f"http://{self.ollama_ip}:{self.ollama_port}/api/generate"
        
        self.mpc_weights = {"Q": [5.0, 5.0, 0.5], "R": [0.5, 0.5]}
        self.convergence_thresholds = {"Q": 0.1, "R": 0.05}
        
        self.system_prompt = """You are an expert in tuning MPC (Model Predictive Control) controllers for autonomous mobile robots (AGVs).
Your task is to analyze the difference between the planned trajectory and the executed trajectory for an AGV and, based on this analysis, suggest new weights for the Q and R matrices of the MPC cost function.

The Q matrix penalizes state deviations (x position error, y position error, theta orientation error). Increasing the values in Q makes the controller more aggressive in correcting trajectory errors.
The R matrix penalizes control effort (linear velocity, angular velocity). Increasing the values in R makes the control smoother and more conservative, saving energy.

**Input Format:**
You will receive a user prompt containing two lists of points: the planned trajectory and the executed trajectory. Each point is in the format [x, y, theta].

**Output Format:**
Your response MUST be a JSON object, and NOTHING ELSE. The JSON must have the following format. It is crucial that Q is a list of exactly 3 float values and R is a list of exactly 2 float values. All returned values for Q and R must be greater than zero.
{
  "Q": [q1, q2, q3],
  "R": [r1, r2]
}

**Analysis Example:**
- If the executed trajectory is too far from the planned one (overshoot), you should increase the weights in Q.
- If the executed trajectory is too oscillatory or unstable, you should increase the weights in R to smooth the control.
- If the orientation error is large, increase the third value of Q.
- Find a balance for precise and smooth navigation.

Analyze the provided data and return only the JSON object with the suggested new weights.
"""

        self.scenario_thread = None
        self.stop_event = threading.Event()
        
        # Inicialização do publisher. Assume que 'self' é uma instância de um Node do ROS 2.
        self.llm_publisher = self.create_publisher(String, 'mpc_weights_recommendation', 10)

    def _get_llm_recommendation(self, agv_id):
        self.get_logger().info(f"\n[LLM] A solicitar recomendação de ajuste de pesos com base no desempenho do {agv_id}...")
        
        previous_weights = self.mpc_weights.copy()
        planned_sample = self.planned_path[::5]
        executed_sample = self.executed_path[::5]

        user_prompt = f"""
Trajectory Analysis for AGV: {agv_id}

Planned Trajectory (sample):
{planned_sample}

Executed Trajectory (sample):
{executed_sample}

Current Weights:
Q = {self.mpc_weights['Q']}
R = {self.mpc_weights['R']}

Based on the comparison of the trajectories, suggest new Q and R weights in the specified JSON format.
"""
        
        payload = {"model": self.llm_model, "system": self.system_prompt, "prompt": user_prompt, "format": "json", "stream": False}

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            response_data = json.loads(response.text)
            new_weights = json.loads(response_data.get("response", "{}"))

            # Validação rigorosa do formato e tipo da resposta do LLM
            if "Q" in new_weights and "R" in new_weights and \
               isinstance(new_weights["Q"], list) and len(new_weights["Q"]) == 3 and \
               isinstance(new_weights["R"], list) and len(new_weights["R"]) == 2:
                
                try:
                    q_as_floats = [float(v) for v in new_weights["Q"]]
                    r_as_floats = [float(v) for v in new_weights["R"]]
                    validated_weights = {"Q": q_as_floats, "R": r_as_floats}
                except (ValueError, TypeError):
                    self.get_logger().warn(f"[LLM] Resposta do LLM continha valores não numéricos: {new_weights}")
                    return

                q_diffs = [abs(a - b) for a, b in zip(validated_weights["Q"], previous_weights["Q"])]
                r_diffs = [abs(a - b) for a, b in zip(validated_weights["R"], previous_weights["R"])]

                q_converged = all(d < self.convergence_thresholds["Q"] for d in q_diffs)
                r_converged = all(d < self.convergence_thresholds["R"] for d in r_diffs)

                if q_converged and r_converged:
                    self.get_logger().info(f"[LLM] CONVERGÊNCIA ATINGIDA! A alteração nos pesos está dentro do limiar de {self.convergence_thresholds}.")
                    self.has_converged = True
                
                self.mpc_weights = validated_weights
                self.get_logger().info(f"[LLM] Novos pesos recebidos: Q={self.mpc_weights['Q']}, R={self.mpc_weights['R']}")
                
                # Enviar os novos pesos para o AGV através do método de sintonia
                agv_instance = next((agv for agv in self.agv_instances if agv.agv_id == agv_id), None)
                if agv_instance and hasattr(agv_instance, 'call_mpc_tuning'):
                    self.get_logger().info(f"A enviar novos pesos Q e R para o AGV {agv_id}...")
                    agv_instance.call_mpc_tuning(Q=self.mpc_weights["Q"], R=self.mpc_weights["R"])
                    self.get_logger().info("A aguardar 1 segundo para o controlador aplicar os ajustes...")
                    time.sleep(1)

                if self.llm_publisher:
                    msg = String()
                    msg.data = json.dumps(self.mpc_weights)
                    self.llm_publisher.publish(msg)
                    self.get_logger().info(f"Publicado no tópico '{self.llm_publisher.topic_name}': {msg.data}")

            else:
                self.get_logger().warn(f"[LLM] Resposta do LLM inválida ou com formato incorreto: {new_weights}")

        except Exception as e:
            self.get_logger().warn(f"[LLM] Erro na comunicação com o Ollama: {e}")

    def _find_lookahead_point(self, agv_instance):
        current_x, current_y, _ = agv_instance.pose
        min_dist_sq = float('inf')
        closest_idx = 0
        for i, (px, py, _) in enumerate(self.planned_path):
            dist_sq = (px - current_x)**2 + (py - current_y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_idx = i
        lookahead_idx = closest_idx
        while lookahead_idx < len(self.planned_path) - 1:
            px, py, _ = self.planned_path[lookahead_idx]
            dist_from_closest_sq = (px - self.planned_path[closest_idx][0])**2 + (py - self.planned_path[closest_idx][1])**2
            if dist_from_closest_sq >= self.lookahead_distance**2: break
            lookahead_idx += 1
        return self.planned_path[lookahead_idx]

    def _scenario_execution_loop(self, target_speed: float = 1.0):
        self.get_logger().info("\n--- A iniciar Cenário 5 (Ciclo de Otimização Contínua com ROS 2) ---")
        
        agv = self.agv_instances[0]
        self.agv_state = 'GOING_FORWARD'
        self.planned_path = self.forward_path
        self.executed_path = []

        while not self.stop_event.is_set() and not self.has_converged:
            current_state = self.agv_state
            
            if current_state == 'GOING_FORWARD':
                end_point = self.planned_path[-1]
                if np.linalg.norm(np.array(agv.pose[:2]) - np.array(end_point[:2])) <= 1.0:
                    self.get_logger().info(f"Percurso de IDA concluído. A iniciar manobra de estacionamento...")
                    self.agv_state = 'INITIATE_PARKING_AT_END'
                else:
                    target_x, target_y, target_theta = self._find_lookahead_point(agv)
                    self.call_mpc(agv, target_x, target_y, target_theta, target_speed)

            elif current_state == 'INITIATE_PARKING_AT_END':
                px, py, p_theta = self.end_parking_spot
                self.get_logger().info(f"A enviar comando único para estacionar em {self.end_parking_spot[:2]}...")
                self.call_mpc(agv, px, py, p_theta, target_speed * 0.7)
                self.agv_state = 'WAITING_FOR_PARKED_AT_END'

            elif current_state == 'WAITING_FOR_PARKED_AT_END':
                px, py, _ = self.end_parking_spot
                if np.linalg.norm(np.array(agv.pose[:2]) - np.array([px, py])) <= 0.2:
                    self.get_logger().info("Estacionado. A chamar LLM para otimização...")
                    self._get_llm_recommendation(agv.agv_id)
                    if self.has_converged: break
                    
                    self.get_logger().info("A iniciar percurso de VOLTA...")
                    self.agv_state = 'GOING_BACKWARD'
                    self.planned_path = self.backward_path
                    self.executed_path.clear()

            elif current_state == 'GOING_BACKWARD':
                end_point = self.planned_path[-1]
                if np.linalg.norm(np.array(agv.pose[:2]) - np.array(end_point[:2])) <= 1.0:
                    self.get_logger().info(f"Percurso de VOLTA concluído. A iniciar manobra de estacionamento...")
                    self.agv_state = 'INITIATE_PARKING_AT_START'
                else:
                    target_x, target_y, target_theta = self._find_lookahead_point(agv)
                    self.call_mpc(agv, target_x, target_y, target_theta, target_speed)

            elif current_state == 'INITIATE_PARKING_AT_START':
                px, py, p_theta = self.start_parking_spot
                self.get_logger().info(f"A enviar comando único para estacionar em {self.start_parking_spot[:2]}...")
                self.call_mpc(agv, px, py, p_theta, target_speed * 0.7)
                self.agv_state = 'WAITING_FOR_PARKED_AT_START'

            elif current_state == 'WAITING_FOR_PARKED_AT_START':
                px, py, _ = self.start_parking_spot
                if np.linalg.norm(np.array(agv.pose[:2]) - np.array([px, py])) <= 0.2:
                    self.get_logger().info("Estacionado. A chamar LLM para otimização...")
                    self._get_llm_recommendation(agv.agv_id)
                    if self.has_converged: break
                    
                    self.get_logger().info("A reiniciar ciclo. A iniciar percurso de IDA...")
                    self.agv_state = 'GOING_FORWARD'
                    self.planned_path = self.forward_path
                    self.executed_path.clear()

            # Apenas armazena a pose se o AGV não estiver em espera
            if 'WAITING' not in current_state:
                self.executed_path.append(agv.pose[:])
            
            time.sleep(0.1)

        self.get_logger().info("--- Loop de execução do Cenário 5 terminado ---")

    def call_mpc(self, agv_instance, x, y, theta, speed):
        if hasattr(agv_instance, 'call_mpc') and callable(agv_instance.call_mpc):
             agv_instance.call_mpc('move', float(x), float(y), float(theta), float(speed))
        else:
             self.get_logger().warn(f"AGV {agv_instance.agv_id} não possui um método 'call_mpc' funcional.")

    def start_scenario(self):
        if not hasattr(self, 'agv_instances') or not self.agv_instances:
            self.get_logger().warn("A lista 'agv_instances' não foi encontrada ou está vazia.")
            return
        
        # Define os caminhos e os pontos de estacionamento com base na posição inicial real do AGV
        start_pose = self.agv_instances[0].pose
        start_y = start_pose[1]
        end_y = 12.5  # O ponto final permanece fixo

        self.forward_path = PathGenerator(self.obstacles, 2.5, start_y, end_y).points
        self.backward_path = PathGenerator(self.obstacles, 2.5, end_y, start_y).points
        
        self.end_parking_spot = (0.0, end_y, -math.pi / 2)
        self.start_parking_spot = (start_pose[0], start_y, start_pose[2])
        
        self.stop_event.clear()
        self.has_converged = False
        self.scenario_thread = threading.Thread(target=self._scenario_execution_loop, args=(1.0,))
        self.scenario_thread.start()
        self.get_logger().info("Thread de gestão do Cenário 5 iniciada.")

    def stop_scenario(self):
        if self.scenario_thread and self.scenario_thread.is_alive():
            self.get_logger().info("\n[CONTROLO] A enviar sinal de paragem para o cenário...")
            self.stop_event.set()
            self.scenario_thread.join()
            self.get_logger().info("[CONTROLO] Cenário parado com sucesso.")