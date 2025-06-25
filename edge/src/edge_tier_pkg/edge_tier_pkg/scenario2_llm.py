import time
import random
import threading
import math
import numpy as np
from collections import deque
import json
import requests

class ProductionToken:
    def __init__(self, order_id, creation_time, logger):
        self.order_id = order_id
        self.creation_time = creation_time 
        self.logger = logger 
        self.line = None
        self.status = "Order_Generated"
        self.events = [] 
        self.current_location = "System_Entry"
        self.total_time = None
        self.processing_start_time = None 
        self._store_event("Order_Generated_System", details=f"Order {order_id} created at {creation_time:.2f}s")

    def _store_event(self, event_type, agv_id=None, time_taken=None, location_from=None, location_to=None, details=""):
        timestamp = time.perf_counter()
        event = {
            "timestamp": timestamp, "event_type": event_type, "agv_id": agv_id,
            "time_taken": time_taken, "location_from": location_from,
            "location_to": location_to, "details": details
        }
        self.events.append(event)

    def log_event(self, event_type, agv_id=None, time_taken=None, location_from=None, location_to=None, details=""):
        self._store_event(event_type, agv_id, time_taken, location_from, location_to, details)

    def mark_material_pickup_from_warehouse(self, event_timestamp=None):
        if self.processing_start_time is None: 
            self.processing_start_time = event_timestamp if event_timestamp is not None else time.perf_counter()
            self._store_event("Order_Processing_Started_From_WH_Pickup", 
                             details=f"Processing clock for {self.order_id} started at WH pickup: {self.processing_start_time:.2f}s")

    def set_line(self, line_id):
        self.line = line_id
        self._store_event("Assigned_To_Line", details=f"Order assigned to Line {line_id}")


    def update_status(self, new_status):
        self.status = new_status

    def get_formatted_event_trace_for_llm(self):
        if not self.events:
            return "No events recorded for this order.\n"
        trace_lines = []
        reference_start_time = self.processing_start_time if self.processing_start_time is not None else \
                               (self.events[0]["timestamp"] if self.events else self.creation_time)
        trace_lines.append(f"--- DETAILED TRACE FOR COMPLETED ORDER: {self.order_id} (Line: {self.line}) ---")
        if self.total_time is not None:
             time_calc_desc = "(from WH pickup)" if self.processing_start_time else "(from order creation)"
             trace_lines.append(f"  Final Status: {self.status}")
             trace_lines.append(f"  Total Processing Time {time_calc_desc}: {self.total_time:.2f}s")
        else:
            trace_lines.append(f"  Final Status: {self.status}")
            trace_lines.append("  Total Processing Time: Not yet finalized for this trace.\n")
        trace_lines.append(f"  Key Events ({len(self.events)} total):\n")
        for event_item in self.events:
            ts = event_item['timestamp']
            time_since_ref_start = ts - reference_start_time
            line_str = f"    - [{time_since_ref_start:6.2f}s rel. to proc. start] Event: {event_item['event_type']}"
            if event_item.get('agv_id'): line_str += f", AGV: {event_item['agv_id']}"
            time_label = ""
            if event_item.get('time_taken') is not None:
                if event_item['event_type'].startswith("Processing_End_"): time_label = " (Station Processing)"
                elif "Loading" in event_item['event_type'] or "Pickup" in event_item['event_type'] or "Unloading" in event_item['event_type']:
                    if not event_item['event_type'].startswith("Arrival_For_Pickup"): time_label = " (Load/Unload)"
                elif ("Arrival" in event_item['event_type'] or "Advance" in event_item['event_type'] or "Returned" in event_item['event_type']):
                    time_label = " (Transport)"
                line_str += f", Duration: {event_item['time_taken']:.2f}s{time_label}"
            if event_item.get('location_from'): line_str += f", From: {event_item['location_from']}"
            if event_item.get('location_to'): line_str += f", To: {event_item['location_to']}"
            if event_item.get('details') and event_item['event_type'] != "Order_Completed_Dispatch_Data": 
                line_str += f", Details: {event_item['details'][:150]}" 
            trace_lines.append(line_str)
        trace_lines.append(f"--- END OF TRACE FOR ORDER: {self.order_id} ---")
        return "\n".join(trace_lines)

    def complete_order(self):
        self.update_status("Completed")
        final_timestamp = time.perf_counter() 
        time_calc_desc = "(from WH pickup)"
        if self.processing_start_time is not None:
            self.total_time = final_timestamp - self.processing_start_time
        else: 
            self.logger.warn(f"[TOKEN {self.order_id}] processing_start_time not set! Using creation time for total_time.")
            self.total_time = final_timestamp - self.creation_time
            time_calc_desc = "(from order creation, fallback)"
        details_msg = f"Total processing time {time_calc_desc}: {self.total_time:.2f}s"
        self._store_event("Order_Completed_Dispatch_Data", time_taken=self.total_time, details=details_msg)


class Scenario2:
    def __init__(self):
        self._logger_instance = None
        self.lock = threading.Lock()
        self.background_threads_lock = threading.Lock() 
        self.active_background_tasks = [] 
        self.simulation_start_time = None 
        
        self.ollama_ip = "192.168.5.112"
        self.ollama_port = 11434 
        self.ollama_model = "llama3.2" 
        
        self.agv_ids_list = [f"agv_{i}" for i in range(1, 9)] 
        self.min_agv_speed = 0.1 
        self.max_agv_speed = 2.0 
        self.agv_target_speeds = {agv_id: 1.0 for agv_id in self.agv_ids_list}
        self.current_target_opm = 0.5 
        self.operator_instructions = ""

        self.llm_system_prompt = self._get_llm_system_prompt()
        self.llm_conversation_history = []
        
        self.positions = {
            'warehouse': [(20.5,0,0),(18.0,0,0),(15.5,0,0),(13.0,0,0)], 'ws1':[(6.5,2,0),(1.5,2,0)],
            'ws2':[(6.5,-2,0),(1.5,-2,0)], 'ws3':[(-1.5,2,0),(-6.5,2,0),(-4,3.5,0)],
            'ws4':[(-1.5,-2,0),(-6.5,-2,0),(-4,-3.5,0)], 'fp':[(-14.5,0,0),(-12,0,0)]
        }
        self.occupancy = {loc:[False]*len(self.positions[loc]) for loc in self.positions}
        self.line_A_token_active = None; self.line_B_token_active = None
        self.ws1_material_delivered=threading.Event(); self.ws1_processed=threading.Event()
        self.ws2_material_delivered=threading.Event(); self.ws2_processed=threading.Event()
        self.ws3_stock_material_delivered=threading.Event(); self.ws3_semifinished_material_delivered=threading.Event()
        self.ws3_processed=threading.Event(); self.ws4_stock_material_delivered=threading.Event()
        self.ws4_semifinished_material_delivered=threading.Event(); self.ws4_processed=threading.Event()
        self.ws1_output_buffer=deque(); self.ws2_output_buffer=deque()
        self.ws3_output_buffer=deque(); self.ws4_output_buffer=deque(); self.fp_delivered_tokens=deque()
        self.ws_processing_times = {'ws1':{'mean':10,'range':4},'ws2':{'mean':10,'range':4},'ws3':{'mean':15,'range':6},'ws4':{'mean':15,'range':6}}
        self.next_order_id = 1; self.next_line_to_assign = 'A'; self.run = False; self.agv_instances = []; self.advancement_delay = 3.0

    def _get_llm_system_prompt(self):
        return f"""You are an AI assistant, an expert in optimizing the rhythm and throughput of an industrial production line.
Your role is to act as an intelligent advisor to a human operator.

Key Definitions:
- OPM (Orders Per Minute): This is a measure of the production line's throughput, indicating how many complete orders are processed by the entire system per minute.
- AGV Speed: This is the physical travel speed of the Automated Guided Vehicles, provided in meters per second (m/s).

You will receive performance data after each completed production order. This data includes:
1. A detailed trace of the completed order: This trace outlines the sequence of operations, AGVs involved, transport durations (how long an AGV took to move), load/unload times, and workstation processing times for that specific order.
2. The current cumulative OPM for the entire system.
3. The operator's current target OPM.
4. Occasionally, a new target OPM or specific instruction from the human operator (this will be explicitly noted if present).

Your primary goal is to help the operator achieve or maintain the **target OPM** by suggesting individual **speed settings** for each of the 8 AGVs ({', '.join(self.agv_ids_list)}).

Constraints and Guidelines:
- AGV speeds must be suggested in meters per second (m/s).
- Each AGV speed must be clamped between a minimum of {self.min_agv_speed:.1f} m/s (to ensure movement) and a maximum of {self.max_agv_speed:.1f} m/s. If your ideal calculation falls outside this range, suggest the closest limit.
- The simulation starts with all AGVs at a default speed (currently {self.agv_target_speeds.get('agv_1', 1.0):.1f} m/s). Your first set of suggestions should consider this baseline.
- Analyze the provided order trace to identify potential bottlenecks or underutilized AGVs. For example, if an AGV has very short transport times followed by long waits, its speed might be too high or other parts of the system are lagging. Conversely, long transport times for critical path AGVs might indicate a need for increased speed.
- Consider the interplay between AGV speeds. Speeding up one AGV might create congestion elsewhere if not balanced.
- The human operator is ultimately in charge and may provide a new target OPM or override suggestions. Always prioritize the latest operator input. If no new operator input is given, continue to adjust speeds based on the latest production data to maintain or achieve the last known target OPM.

Output Format for Speed Suggestions:
When suggesting speed changes, YOU MUST provide them in a specific JSON-like dictionary format embedded in your response, clearly marked:
`SPEED_SUGGESTIONS: {{ "agv_1": 1.2, "agv_2": 0.8, ... , "agv_8": 1.0 }}`
Only include the `SPEED_SUGGESTIONS:` block if you are actually suggesting changes. If no changes are needed (e.g., the system is perfectly on target and balanced), you can state that and omit the block.
You should also provide a brief (1-2 sentences) text explanation or reasoning for your suggestions or observations. Your full response will be shown to the operator.
The simulation can only parse speeds from the `SPEED_SUGGESTIONS:` block.
"""

    def get_logger(self):
        if self._logger_instance is None:
            class StandaloneLogger:
                def info(self, msg): print(f"{msg}") 
                def warn(self, msg): print(f"WARN: {msg}")
                def error(self, msg, exc_info=False):
                    print(f"ERROR: {msg}")
                    if exc_info:
                        import traceback, sys
                        if isinstance(exc_info, Exception): traceback.print_exception(type(exc_info), exc_info, exc_info.__traceback__)
                        else: traceback.print_exception(*sys.exc_info())
            self._logger_instance = StandaloneLogger()
        return self._logger_instance

    def _communicate_with_llm(self, current_user_prompt_text):
        messages_to_send = self.llm_conversation_history + [{"role": "user", "content": current_user_prompt_text}]

        try:
            payload = {
                "model": self.ollama_model, "messages": messages_to_send, "stream": False,
                "options": {"temperature": 0.3}
            }
            response = requests.post(
                f"http://{self.ollama_ip}:{self.ollama_port}/api/chat", 
                json=payload, timeout=120 )
            response.raise_for_status() 
            response_data = response.json()
            llm_response_content = response_data.get("message", {}).get("content", "")
            self.llm_conversation_history.append({"role": "user", "content": current_user_prompt_text})
            self.llm_conversation_history.append({"role": "assistant", "content": llm_response_content})
            max_exchanges = 5 
            if len(self.llm_conversation_history) > (1 + max_exchanges * 2): 
                self.llm_conversation_history = [self.llm_conversation_history[0]] + \
                                                self.llm_conversation_history[-(max_exchanges*2):]
            return llm_response_content
        except requests.exceptions.Timeout:
            self.get_logger().error(f"LLM communication error: Timeout after 120 seconds.")
            return "Error: LLM request timed out."
        except requests.exceptions.ConnectionError:
            self.get_logger().error(f"LLM communication error: Connection refused or failed. Is Ollama running at {self.ollama_ip}:{self.ollama_port}?")
            return f"Error: Could not connect to LLM at {self.ollama_ip}:{self.ollama_port}."
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"LLM communication error: {e}")
            return f"Error: LLM communication failed. {e}"
        except Exception as e: 
            self.get_logger().error(f"Error processing LLM response: {e}", exc_info=True)
            return f"Error: Could not process LLM response. {e}"

    def _parse_and_apply_llm_speed_suggestions(self, llm_response_content):
        if not llm_response_content or llm_response_content.startswith("Error:"):
            self.get_logger().warn(f"Skipping LLM suggestion parsing due to previous error or empty response: {llm_response_content}")
            return
        self.get_logger().info(f"[FROM LLM ({self.ollama_model}) - Full Response]:\n{llm_response_content}\n---")
        try:
            speed_suggestions_marker = "SPEED_SUGGESTIONS:"
            if speed_suggestions_marker in llm_response_content:
                suggestions_str_part = llm_response_content.split(speed_suggestions_marker, 1)[1].strip()
                dict_start = suggestions_str_part.find('{')
                dict_end = suggestions_str_part.rfind('}') + 1
                if dict_start != -1 and dict_end > dict_start :
                    suggestions_dict_str = suggestions_str_part[dict_start:dict_end]
                    suggested_speeds = json.loads(suggestions_dict_str)
                    changes_applied = False
                    for agv_id, speed in suggested_speeds.items():
                        if agv_id in self.agv_target_speeds:
                            try:
                                new_speed = float(speed)
                                clamped_speed = round(max(self.min_agv_speed, min(self.max_agv_speed, new_speed)), 2)
                                if self.agv_target_speeds[agv_id] != clamped_speed:
                                    self.get_logger().info(f"Applying LLM suggestion: AGV {agv_id} target speed updated from {self.agv_target_speeds[agv_id]:.2f} to {clamped_speed:.2f} m/s.")
                                    self.agv_target_speeds[agv_id] = clamped_speed
                                    agv_instance = self.get_agv_instance(agv_id)
                                    agv_instance.target_velocity = clamped_speed
                                    changes_applied = True
                            except ValueError:
                                self.get_logger().warn(f"LLM suggested invalid speed format for {agv_id}: '{speed}'")
                        else:
                            self.get_logger().warn(f"LLM suggested speed for unknown AGV ID: {agv_id}")
                    if not changes_applied and suggested_speeds:
                         self.get_logger().info("LLM provided speed suggestions, but they resulted in no change to current target speeds.")
                else:
                    self.get_logger().warn(f"Could not parse SPEED_SUGGESTIONS dictionary from LLM response part: '{suggestions_str_part}'")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Error decoding JSON from LLM speed suggestions: {e}. String part: '{suggestions_str_part if 'suggestions_str_part' in locals() else 'N/A'}'")
        except Exception as e:
            self.get_logger().error(f"Unexpected error applying LLM speed suggestions: {e}", exc_info=True)

    def _parse_opm_from_instruction(self, instruction_str, current_opm_target):
        try:
            import re
            match = re.search(r"(?:target OPM|set OPM to|OPM of)\s*([0-9\.]+)", instruction_str, re.IGNORECASE)
            if match:
                new_opm = float(match.group(1))
                return new_opm
        except ValueError:
            self.get_logger().warn(f"Could not parse OPM value from operator instruction: '{instruction_str}'")
        return current_opm_target 

    def _perform_llm_priming_in_background(self):
        try:
            priming_user_prompt = f"System initiating. Current target OPM is {self.current_target_opm:.2f}. Default AGV speeds are {self.agv_target_speeds.get('agv_1', 1.0):.1f} m/s. Awaiting first order completion data. Please acknowledge readiness based on system prompt."
            initial_llm_response = self._communicate_with_llm(priming_user_prompt)
            self._parse_and_apply_llm_speed_suggestions(initial_llm_response) 
        except Exception as e:
            self.get_logger().error(f"Exception in LLM priming thread: {e}", exc_info=True)
        finally:
            with self.background_threads_lock:
                 if threading.current_thread() in self.active_background_tasks:
                    self.active_background_tasks.remove(threading.current_thread())

    def _execute_llm_order_analysis_task(self, completed_token, opm_info_for_llm, detailed_order_trace_for_llm):
        try:
            if not self.run:
                self.get_logger().warn(f"LLM order analysis for {completed_token.order_id} aborted, scenario stopped.")
                return

            target_opm_info_for_llm = f"The current target OPM set by the operator is: {self.current_target_opm:.2f}."
            operator_input_text_for_this_call = ""
            
            if self.operator_instructions: 
                operator_input_text_for_this_call = f"\nNEW OPERATOR INSTRUCTION: \"{self.operator_instructions}\""
                parsed_opm = self._parse_opm_from_instruction(self.operator_instructions, self.current_target_opm)
                if parsed_opm != self.current_target_opm:
                     self.current_target_opm = parsed_opm
                     target_opm_info_for_llm = f"The NEW target OPM set by the operator is: {self.current_target_opm:.2f}." 
                self.operator_instructions = ""

            current_speeds_text = f"Current AGV target speeds (m/s): {json.dumps(self.agv_target_speeds)}"
            llm_user_prompt = (
                f"Production Data Update after completion of Order {completed_token.order_id}:\n"
                f"{opm_info_for_llm}\n" 
                f"{target_opm_info_for_llm}\n"
                f"{operator_input_text_for_this_call}\n"
                f"\n{detailed_order_trace_for_llm}\n" 
                f"\n{current_speeds_text}\n\n"
                f"Based on all the above, your role, and the constraints, provide your analysis and any speed adjustment suggestions (using the SPEED_SUGGESTIONS format)."
            )
            llm_response = self._communicate_with_llm(llm_user_prompt)
            self._parse_and_apply_llm_speed_suggestions(llm_response)
        except Exception as e:
            token_id_log = completed_token.order_id if completed_token else "Unknown_Token"
            self.get_logger().error(f"Exception in _execute_llm_order_analysis_task for (T: {token_id_log}): {e}", exc_info=True)
        finally:
            with self.background_threads_lock:
                if threading.current_thread() in self.active_background_tasks:
                    self.active_background_tasks.remove(threading.current_thread())

    def _execute_adhoc_llm_interaction_task(self, operator_command_text: str):
        """Executa uma interação ad-hoc com o LLM baseada no comando do operador."""
        try:
            if not self.run:
                self.get_logger().warn(f"Ad-hoc LLM interaction ('{operator_command_text[:50]}...') aborted, scenario not running.")
                return

            self.get_logger().info(f"Processing ad-hoc operator command for LLM: '{operator_command_text}'")

            parsed_opm = self._parse_opm_from_instruction(operator_command_text, self.current_target_opm)
            if parsed_opm != self.current_target_opm:
                self.current_target_opm = parsed_opm

            target_opm_info_for_llm = f"The current target OPM is: {self.current_target_opm:.2f} (may have been just updated by your command)."
            current_speeds_text = f"Current AGV target speeds (m/s): {json.dumps(self.agv_target_speeds)}"

            llm_user_prompt = (
                f"ADHOC OPERATOR COMMAND/QUERY:\n"
                f"\"{operator_command_text}\"\n\n"
                f"Current System Context:\n"
                f"- {target_opm_info_for_llm}\n"
                f"- {current_speeds_text}\n\n"
                f"Based on this command and current context, provide your analysis and any speed adjustment suggestions (using the SPEED_SUGGESTIONS format)."
            )
            
            llm_response = self._communicate_with_llm(llm_user_prompt)
            self._parse_and_apply_llm_speed_suggestions(llm_response)
            
            self.get_logger().info(f"Ad-hoc LLM interaction task for command '{operator_command_text[:50]}...' complete.")

        except Exception as e:
            self.get_logger().error(f"Exception in _execute_adhoc_llm_interaction_task for command '{operator_command_text[:50]}...': {e}", exc_info=True)
        finally:
            with self.background_threads_lock:
                if threading.current_thread() in self.active_background_tasks:
                    self.active_background_tasks.remove(threading.current_thread())


    def send_operator_command_to_llm(self, command_text: str):
        """
        Permite que o operador envie um comando/mensagem para o LLM a qualquer momento.
        A interação com o LLM ocorrerá em uma thread de background.
        """
        if not self.run:
            self.get_logger().warn("Cannot send operator command to LLM: Scenario not running.")
            return
        if not command_text or not command_text.strip():
            self.get_logger().warn("Cannot send empty operator command to LLM.")
            return

        self.get_logger().info(f"Operator command received: '{command_text}'. Dispatching to LLM in background.")
        
        adhoc_llm_thread = threading.Thread(target=self._execute_adhoc_llm_interaction_task, args=(command_text,))
        adhoc_llm_thread.daemon = True
        with self.background_threads_lock:
            self.active_background_tasks.append(adhoc_llm_thread)
        adhoc_llm_thread.start()


    def _log_cumulative_opm_and_interact_with_llm(self, completed_token):
        if not hasattr(self, 'simulation_start_time') or self.simulation_start_time is None:
            self.get_logger().warn("Simulation start time not set. Cannot log OPM or interact with LLM.")
            return 
        completed_orders_count = len(self.fp_delivered_tokens)
        if completed_orders_count == 0: return
        current_time = time.perf_counter()
        elapsed_seconds = current_time - self.simulation_start_time
        if elapsed_seconds < 0.1: return
        elapsed_minutes = elapsed_seconds / 60.0
        cumulative_opm_value = completed_orders_count / elapsed_minutes
        
        opm_message_for_log = f"[PERFORMANCE] Order {completed_token.order_id} completed. Cumulative OPM: {cumulative_opm_value:.2f} (Orders: {completed_orders_count}, Total Simulation Time: {elapsed_minutes:.2f} min)"
        self.get_logger().info(opm_message_for_log) 

        detailed_order_trace_string_for_llm = completed_token.get_formatted_event_trace_for_llm()
        
        if self.run: 
            opm_info_for_llm_str = f"Current Cumulative Orders Per Minute (OPM): {cumulative_opm_value:.2f}. Total orders completed: {completed_orders_count}. Total simulation time: {elapsed_minutes:.2f} minutes."
            llm_analysis_args = {
                'completed_token': completed_token,
                'opm_info_for_llm': opm_info_for_llm_str,
                'detailed_order_trace_for_llm': detailed_order_trace_string_for_llm
            }
            llm_thread = threading.Thread(target=self._execute_llm_order_analysis_task, kwargs=llm_analysis_args)
            llm_thread.daemon = True
            with self.background_threads_lock:
                self.active_background_tasks.append(llm_thread)
            llm_thread.start()

    def get_agv_instance(self, agv_id):
        for agv_instance in self.agv_instances:
            if agv_instance.agv_id == agv_id: return agv_instance
        self.get_logger().error(f"AGV instance {agv_id} not found!")
        return None

    def call_mpc(self, agv_instance, x, y, theta, speed):
        if hasattr(agv_instance, 'call_mpc') and callable(agv_instance.call_mpc):
             agv_instance.call_mpc('move', float(x), float(y), float(theta), float(speed))
        else:
             self.get_logger().warn(f"AGV {agv_instance.agv_id} does not have a callable 'call_mpc' method that accepts speed.")


    def calc_dist(self, pose, ref_tuple): 
        if not (isinstance(pose, (list, tuple)) and len(pose) >= 2 and
                isinstance(ref_tuple, (list, tuple)) and len(ref_tuple) >=3 ):
            self.get_logger().error(f"Invalid pose/reference tuple for calc_dist. Pose: {pose}, Ref: {ref_tuple}")
            return float('inf')
        target_x, target_y, _ = ref_tuple
        return np.linalg.norm(np.array(pose[:2]) - np.array([target_x, target_y]))

    def _execute_agv_move_task(self, agv_instance, target_coords_tuple, token, task_description):
        x, y, theta_deg = target_coords_tuple
        agv_id = agv_instance.agv_id 
        target_speed = self.agv_target_speeds.get(agv_id, 1.0) 
        
        start_time = time.perf_counter()
        if not self.run: 
            self.get_logger().warn(f"Movement for AGV {agv_id} to {task_description} aborted (scenario not running).")
            return -1
        self.call_mpc(agv_instance, x, y, theta_deg, target_speed) 
        arrival_threshold = 0.3; polling_interval = 0.1; max_wait_time = 60
        wait_start_time = time.perf_counter()
        while True:
            if not self.run: 
                self.get_logger().warn(f"Movement for AGV {agv_id} to {task_description} interrupted. Dist: {self.calc_dist(agv_instance.pose, target_coords_tuple):.2f}.")
                return -1 
            distance = self.calc_dist(agv_instance.pose, target_coords_tuple)
            if distance <= arrival_threshold: break
            if (time.perf_counter() - wait_start_time > max_wait_time):
                self.get_logger().warn(f"Movement for AGV {agv_id} to {task_description} TIMEOUT. Dist: {distance:.2f}.")
                return -1 
            time.sleep(polling_interval)
        return time.perf_counter() - start_time 

    def _generic_agv_move_and_occupy(self, agv_instance, token, target_loc_name, target_slot_idx, task_name_suffix, occupy_slot=True):
        if target_loc_name not in self.positions or not (0 <= target_slot_idx < len(self.positions[target_loc_name])):
            self.get_logger().error(f"Invalid target '{target_loc_name}[{target_slot_idx}]' for AGV {agv_instance.agv_id}.")
            return -1
        log_token_id = token.order_id if token else "N/A"
        if occupy_slot:
            acquired_slot = False
            while self.run and not acquired_slot:
                with self.lock:
                    if not self.occupancy[target_loc_name][target_slot_idx]:
                        self.occupancy[target_loc_name][target_slot_idx] = True
                        acquired_slot = True; break
                if not acquired_slot: time.sleep(0.5)
            if not self.run or not acquired_slot:
                self.get_logger().warn(f"AGV {agv_instance.agv_id} (T: {log_token_id}) failed to occupy {target_loc_name}[{target_slot_idx}] for {task_name_suffix}.")
                return -1
        
        target_pos_tuple_for_move = self.positions[target_loc_name][target_slot_idx]
        transport_time = self._execute_agv_move_task(agv_instance, target_pos_tuple_for_move, token, f"{task_name_suffix} to {target_loc_name}[{target_slot_idx}]")
        if transport_time < 0 :
             self.get_logger().error(f"AGV {agv_instance.agv_id} movement failed for {task_name_suffix} to {target_loc_name}[{target_slot_idx}].")
             if occupy_slot:
                 with self.lock: self.occupancy[target_loc_name][target_slot_idx] = False
             return -1
        return transport_time

    def _simulate_operation(self, operation_name, duration_mean, duration_range, token, agv_id=None):
        op_duration = random.uniform(duration_mean - duration_range, duration_mean + duration_range)
        if op_duration < 0: op_duration = 0.1
        end_sleep_time = time.perf_counter() + op_duration
        start_op_time = time.perf_counter()
        while time.perf_counter() < end_sleep_time:
            if not self.run:
                self.get_logger().warn(f"Operation '{operation_name}' (T:{token.order_id if token else 'N/A'}) interrupted.")
                return time.perf_counter() - start_op_time 
            time.sleep(min(0.1, max(0, end_sleep_time - time.perf_counter())))
        return op_duration
    
    def _execute_agv_return_to_origin_task(self, agv_instance, token, return_to_station_name, return_to_slot_idx, initial_location_log, task_prefix=""):
        try:
            log_token_id = token.order_id if token else "N/A" 
            if not self.run:
                self.get_logger().warn(f"AGV {agv_instance.agv_id} (Background - T: {log_token_id}) return aborted before move.")
                return
            return_target_loc_log = f"{return_to_station_name}[{return_to_slot_idx}]"
            log_event_prefix = f"{task_prefix}_Returned_To_Idle" if task_prefix else "Returned_To_Idle_Origin"
            
            time_to_return = self._generic_agv_move_and_occupy(
                agv_instance, token, return_to_station_name, return_to_slot_idx,
                f"{task_prefix}_BG_Return_To_{return_target_loc_log.replace('[','_').replace(']','')}",
                occupy_slot=True )

            if time_to_return < 0:
                self.get_logger().error(f"AGV {agv_instance.agv_id} (Background - T: {log_token_id}) failed {task_prefix} return to {return_target_loc_log}.")
            else:
                if token: 
                    token._store_event(log_event_prefix, agv_id=agv_instance.agv_id, time_taken=time_to_return,
                                    location_from=initial_location_log, 
                                    location_to=return_target_loc_log)
                with self.lock:
                    if return_to_station_name in self.occupancy and 0 <= return_to_slot_idx < len(self.occupancy[return_to_station_name]):
                        self.occupancy[return_to_station_name][return_to_slot_idx] = False
                    else: self.get_logger().error(f"AGV {agv_instance.agv_id} (Background - T: {log_token_id}) could not free invalid parking slot {return_target_loc_log}.")
        except Exception as e:
            agv_id_log = agv_instance.agv_id if agv_instance else "Unknown_AGV"; token_id_log = token.order_id if token else "Unknown_Token"
            self.get_logger().error(f"Exception in _execute_agv_return_to_origin_task for AGV {agv_id_log} (T: {token_id_log}): {e}", exc_info=e)
        finally:
            with self.background_threads_lock:
                if threading.current_thread() in self.active_background_tasks:
                    self.active_background_tasks.remove(threading.current_thread())

    def agv_warehouse_pickup_and_deliver(self, agv_id, token, destination_station_name, destination_slot_idx):
        agv_instance = self.get_agv_instance(agv_id);
        if not agv_instance: self.get_logger().error(f"AGV {agv_id} not found (T: {token.order_id})."); return
        token.update_status(f"AGV_{agv_id}_To_Warehouse")
        current_wh_slot_idx = -1; acquired_initial_slot = False
        while self.run and not acquired_initial_slot:
            best_candidate_idx = -1; min_dist_to_slot = float('inf')
            with self.lock:
                current_agv_pose_xy = agv_instance.pose[:2]
                for i in range(len(self.positions['warehouse'])):
                    if not self.occupancy['warehouse'][i]:
                        slot_pos_xy = self.positions['warehouse'][i][:2]
                        dist = math.sqrt((current_agv_pose_xy[0] - slot_pos_xy[0])**2 + (current_agv_pose_xy[1] - slot_pos_xy[1])**2)
                        if dist < min_dist_to_slot: min_dist_to_slot = dist; best_candidate_idx = i
                if best_candidate_idx != -1:
                    self.occupancy['warehouse'][best_candidate_idx] = True; current_wh_slot_idx = best_candidate_idx; acquired_initial_slot = True
            if not acquired_initial_slot and self.run: time.sleep(0.5)
        if not self.run or not acquired_initial_slot: self.get_logger().warn(f"AGV {agv_id} (T: {token.order_id}) task aborted. Failed to acquire WH slot."); return
        initial_agv_pose_for_log = agv_instance.pose
        target_wh_coords = self.positions['warehouse'][current_wh_slot_idx]
        transport_time = self._execute_agv_move_task(agv_instance, target_wh_coords, token, f"Move_To_WH_Slot_{current_wh_slot_idx}")
        if transport_time < 0: 
            self.get_logger().error(f"AGV {agv_id} (T: {token.order_id}) failed move to WH slot {current_wh_slot_idx}.");
            with self.lock: self.occupancy['warehouse'][current_wh_slot_idx] = False; return
        token.log_event("Warehouse_Slot_Arrival", agv_id=agv_id, time_taken=transport_time, location_from=f"Pose({initial_agv_pose_for_log[0]:.1f},{initial_agv_pose_for_log[1]:.1f})", location_to=f"WH_Slot_{current_wh_slot_idx}")
        
        while self.run and current_wh_slot_idx > 0:
            next_target_idx = current_wh_slot_idx - 1; can_advance = False
            with self.lock:
                if not self.occupancy['warehouse'][next_target_idx]: can_advance = True
            if can_advance:
                _ = self._simulate_operation(f"PreAdvDelay_WH_{current_wh_slot_idx}to{next_target_idx}", self.advancement_delay, 0, token, agv_id=agv_id)
                if not self.run: break 
                moved = False
                with self.lock:
                    if not self.occupancy['warehouse'][next_target_idx]:
                        self.occupancy['warehouse'][next_target_idx] = True; self.occupancy['warehouse'][current_wh_slot_idx] = False
                        slot_left = current_wh_slot_idx; current_wh_slot_idx = next_target_idx; moved = True
                if moved:
                    adv_time = self._execute_agv_move_task(agv_instance, self.positions['warehouse'][current_wh_slot_idx], token, f"Advance_To_WH_Slot_{current_wh_slot_idx}")
                    if adv_time < 0: 
                        self.get_logger().error(f"AGV {agv_id} (T: {token.order_id}) failed move to WH {current_wh_slot_idx} after delay. Reverting.")
                        with self.lock: self.occupancy['warehouse'][current_wh_slot_idx] = False; current_wh_slot_idx = slot_left; self.occupancy['warehouse'][current_wh_slot_idx] = True
                    else: token.log_event("Warehouse_Slot_Advance", agv_id=agv_id, time_taken=adv_time, location_from=f"WH_Slot_{slot_left}", location_to=f"WH_Slot_{current_wh_slot_idx}")
            else: time.sleep(0.2)
        
        if not self.run or current_wh_slot_idx != 0: 
            log_msg = f"AGV {agv_id} (T: {token.order_id}) "
            if current_wh_slot_idx !=0 : log_msg += f"failed to reach WH slot 0. Stuck at {current_wh_slot_idx}."
            else: log_msg += "aborted during WH advance to slot 0."
            self.get_logger().error(log_msg) if current_wh_slot_idx !=0 else self.get_logger().warn(log_msg)
            if 0 <= current_wh_slot_idx < len(self.positions['warehouse']):
                with self.lock: self.occupancy['warehouse'][current_wh_slot_idx] = False
            return
        
        token.log_event("Warehouse_Pickup_Ready", agv_id=agv_id, location_to="WH_PickupPoint_Slot0")
        load_time = self._simulate_operation("Material_Loading_Warehouse", 1.5, 0.5, token, agv_id)
        event_pickup_timestamp = time.perf_counter() 
        token.log_event("Material_Pickup_Warehouse", agv_id=agv_id, time_taken=load_time, location_from="WH_PickupPoint_Slot0")
        token.mark_material_pickup_from_warehouse(event_timestamp=event_pickup_timestamp) 
        
        with self.lock: self.occupancy['warehouse'][0] = False
        
        token.update_status(f"AGV_{agv_id}_Delivering_To_{destination_station_name}")
        from_loc = f"WH_Slot0(Pose:{agv_instance.pose[0]:.1f},{agv_instance.pose[1]:.1f})"
        delivery_time = self._generic_agv_move_and_occupy(agv_instance, token, destination_station_name, destination_slot_idx, "Deliver_Material")
        if delivery_time < 0 : self.get_logger().error(f"AGV {agv_id} (T: {token.order_id}) failed delivery to {destination_station_name}[{destination_slot_idx}]."); return
        token.log_event("Material_Delivery_Workstation", agv_id=agv_id, time_taken=delivery_time, location_from=from_loc, location_to=f"{destination_station_name}[{destination_slot_idx}]")

    def agv_transfer_station_to_station(self, agv_id, token, from_station_name, from_station_pickup_slot_idx, to_station_name, to_station_delivery_slot_idx, return_to_station_name, return_to_slot_idx, item_description="SemiFinished"):
        agv_instance = self.get_agv_instance(agv_id); 
        if not agv_instance: return
        token.update_status(f"AGV_{agv_id}_To_Pickup_{item_description}_At_{from_station_name}")
        pickup_loc = f"{from_station_name}[{from_station_pickup_slot_idx}]"
        time_to_pickup = self._generic_agv_move_and_occupy(agv_instance, token, from_station_name, from_station_pickup_slot_idx, f"Move_For_Pickup_{item_description}")
        if time_to_pickup < 0: self.get_logger().error(f"AGV {agv_id} (T: {token.order_id}) failed pickup at {pickup_loc}."); return
        token.log_event(f"Arrival_For_Pickup_{item_description}", agv_id=agv_id, time_taken=time_to_pickup, location_to=pickup_loc)
        load_time = self._simulate_operation(f"{item_description}_Loading", 1.5, 0.5, token, agv_id=agv_id)
        token.log_event(f"{item_description}_Pickup", agv_id=agv_id, time_taken=load_time, location_from=pickup_loc)
        with self.lock: self.occupancy[from_station_name][from_station_pickup_slot_idx] = False
        
        token.update_status(f"AGV_{agv_id}_Delivering_{item_description}_To_{to_station_name}")
        delivery_loc = f"{to_station_name}[{to_station_delivery_slot_idx}]"
        time_to_delivery = self._generic_agv_move_and_occupy(agv_instance, token, to_station_name, to_station_delivery_slot_idx, f"Deliver_{item_description}")
        if time_to_delivery < 0: self.get_logger().error(f"AGV {agv_id} (T: {token.order_id}) failed delivery at {delivery_loc}."); return 
        token.log_event(f"{item_description}_Delivery", agv_id=agv_id, time_taken=time_to_delivery, location_from=pickup_loc, location_to=delivery_loc)
        
        _ = self._simulate_operation(f"Parking_At_{delivery_loc.replace('[','_').replace(']','')}", 2.0, 0, token, agv_id=agv_id)
        if not self.run : self.get_logger().warn(f"AGV {agv_id} (T: {token.order_id}) return aborted during parking."); return 

        loc_after_delivery = f"Pose({agv_instance.pose[0]:.1f},{agv_instance.pose[1]:.1f})" 
        if self.run:
            return_args = {'agv_instance': agv_instance, 'token': token, 'return_to_station_name': return_to_station_name, 
                           'return_to_slot_idx': return_to_slot_idx, 'initial_location_log': loc_after_delivery, 'task_prefix': "SF_Transfer"}
            return_thread = threading.Thread(target=self._execute_agv_return_to_origin_task, kwargs=return_args)
            return_thread.daemon = True 
            with self.background_threads_lock: self.active_background_tasks.append(return_thread)
            return_thread.start()

    def agv_pickup_and_deliver_to_dispatch(self, agv_id, token, from_workstation_name, from_workstation_pickup_slot_idx):
        agv_instance = self.get_agv_instance(agv_id);
        if not agv_instance: self.get_logger().error(f"AGV {agv_id} not found for T: {token.order_id if token else 'N/A'}."); return
        token.update_status(f"AGV_{agv_id}_To_Pickup_FP_At_{from_workstation_name}")
        pickup_loc = f"{from_workstation_name}[{from_workstation_pickup_slot_idx}]"
        time_to_pickup = self._generic_agv_move_and_occupy(agv_instance, token, from_workstation_name, from_workstation_pickup_slot_idx, "Move_For_Pickup_FP")
        if time_to_pickup < 0: self.get_logger().error(f"AGV {agv_id} (T:{token.order_id}) failed pickup at {pickup_loc}."); return
        token.log_event("Arrival_For_Pickup_FP", agv_id=agv_id, time_taken=time_to_pickup, location_to=pickup_loc)
        load_time = self._simulate_operation("FP_Loading", 1.5, 0.5, token, agv_id=agv_id)
        token.log_event("FP_Pickup", agv_id=agv_id, time_taken=load_time, location_from=pickup_loc)
        with self.lock: self.occupancy[from_workstation_name][from_workstation_pickup_slot_idx] = False
        
        token.update_status(f"AGV_{agv_id}_To_Dispatch")
        current_fp_slot_idx = -1; acquired_fp_slot = False; preferred_fp_slot_order = [0, 1]
        while self.run and not acquired_fp_slot:
            with self.lock:
                for slot_idx in preferred_fp_slot_order:
                    if 0 <= slot_idx < len(self.positions['fp']) and not self.occupancy['fp'][slot_idx]:
                        self.occupancy['fp'][slot_idx] = True; current_fp_slot_idx = slot_idx; acquired_fp_slot = True; break
            if not acquired_fp_slot and self.run: time.sleep(0.5)
        if not self.run or not acquired_fp_slot: self.get_logger().warn(f"AGV {agv_id} (T:{token.order_id}) aborted. Failed to acquire dispatch slot."); return
        
        time_to_fp_slot = self._execute_agv_move_task(agv_instance, self.positions['fp'][current_fp_slot_idx], token, f"Move_To_Dispatch_Slot_{current_fp_slot_idx}")
        if time_to_fp_slot < 0:
            self.get_logger().error(f"AGV {agv_id} (T:{token.order_id}) failed move to dispatch slot {current_fp_slot_idx}.");
            with self.lock: self.occupancy['fp'][current_fp_slot_idx] = False; return
        token.log_event("Dispatch_Slot_Arrival", agv_id=agv_id, time_taken=time_to_fp_slot, location_from=pickup_loc, location_to=f"Dispatch_Slot_{current_fp_slot_idx}")
        
        while self.run and current_fp_slot_idx > 0: 
            next_target_idx = current_fp_slot_idx - 1; can_advance = False
            with self.lock:
                if not self.occupancy['fp'][next_target_idx]: can_advance = True
            if can_advance:
                _ = self._simulate_operation(f"PreAdvDelay_FP_{current_fp_slot_idx}to{next_target_idx}", self.advancement_delay, 0, token, agv_id=agv_id)
                if not self.run: break
                moved = False
                with self.lock:
                    if not self.occupancy['fp'][next_target_idx]:
                        self.occupancy['fp'][next_target_idx] = True; self.occupancy['fp'][current_fp_slot_idx] = False
                        slot_left = current_fp_slot_idx; current_fp_slot_idx = next_target_idx; moved = True
                if moved:
                    adv_time = self._execute_agv_move_task(agv_instance, self.positions['fp'][current_fp_slot_idx], token, f"Advance_To_Dispatch_Slot_{current_fp_slot_idx}")
                    if adv_time < 0:
                        self.get_logger().error(f"AGV {agv_id} (T:{token.order_id}) failed move to FP {current_fp_slot_idx} after delay. Reverting.")
                        with self.lock: self.occupancy['fp'][current_fp_slot_idx] = False; current_fp_slot_idx = slot_left; self.occupancy['fp'][current_fp_slot_idx] = True
                    else: token.log_event("Dispatch_Slot_Advance", agv_id=agv_id, time_taken=adv_time, location_from=f"Dispatch_Slot_{slot_left}", location_to=f"Dispatch_Slot_{current_fp_slot_idx}")
            else: time.sleep(0.2)

        if not self.run or current_fp_slot_idx != 0:
            log_msg = f"AGV {agv_id} (T:{token.order_id}) "
            if current_fp_slot_idx !=0: log_msg += f"failed to reach dispatch slot 0. Stuck at {current_fp_slot_idx}."
            else: log_msg += f"aborted during dispatch advance to slot 0."
            self.get_logger().error(log_msg) if current_fp_slot_idx !=0 else self.get_logger().warn(log_msg)
            if 0 <= current_fp_slot_idx < len(self.positions['fp']): 
                with self.lock: self.occupancy['fp'][current_fp_slot_idx] = False
            return
        
        token.log_event("Dispatch_Delivery_Ready", agv_id=agv_id, location_to="Dispatch_Point_Slot0")
        unload_time = self._simulate_operation("FP_Unloading_Dispatch", 1.5, 0.5, token, agv_id=agv_id)
        
        token.complete_order() 
        self.fp_delivered_tokens.append(token)
        self._log_cumulative_opm_and_interact_with_llm(token) 
        
        dispatch_pose_log = f"Pose({agv_instance.pose[0]:.1f},{agv_instance.pose[1]:.1f})"
        with self.lock: self.occupancy['fp'][0] = False 
        
        if self.run:
            return_args = {'agv_instance': agv_instance, 'token': token, 'return_to_station_name': from_workstation_name, 
                           'return_to_slot_idx': from_workstation_pickup_slot_idx, 'initial_location_log': dispatch_pose_log, 'task_prefix': "Dispatch"}
            return_thread = threading.Thread(target=self._execute_agv_return_to_origin_task, kwargs=return_args)
            return_thread.daemon = True
            with self.background_threads_lock: self.active_background_tasks.append(return_thread)
            return_thread.start()

    def _process_at_workstation(self, ws_name, token, input_slots_indices_to_free, output_buffer_deque, processed_event_to_set):
        ws_settings = self.ws_processing_times[ws_name]
        token.update_status(f"Processing_At_{ws_name}")
        token.log_event(f"Processing_Start_{ws_name.upper()}", location_to=ws_name, details=f"Using inputs from slots {input_slots_indices_to_free}")
        processing_time = self._simulate_operation(f"Processing_{ws_name.upper()}", ws_settings['mean'], ws_settings['range'], token)
        token.log_event(f"Processing_End_{ws_name.upper()}", time_taken=processing_time, location_from=ws_name)
        with self.lock:
            for slot_idx in input_slots_indices_to_free:
                if ws_name in self.occupancy and 0 <= slot_idx < len(self.occupancy[ws_name]):
                    if self.occupancy[ws_name][slot_idx]: self.occupancy[ws_name][slot_idx] = False
                    else: self.get_logger().warn(f"Token {token.order_id}: WS {ws_name} input slot {slot_idx} was ALREADY free.")
                else: self.get_logger().error(f"Token {token.order_id}: Invalid workstation '{ws_name}' or slot '{slot_idx}' for freeing.")
        if output_buffer_deque is not None: output_buffer_deque.append(token)
        if processed_event_to_set is not None: processed_event_to_set.set()

    def run_production_line_A(self, token):
        current_token = token 
        try:
            self.line_A_token_active = current_token; current_token.set_line('A')
            agv1_thread = threading.Thread(target=self.agv_warehouse_pickup_and_deliver, args=('agv_1', current_token, 'ws1', 0))
            agv5_thread = threading.Thread(target=self.agv_warehouse_pickup_and_deliver, args=('agv_5', current_token, 'ws3', 2))
            agv1_thread.start(); agv5_thread.start()
            agv1_thread.join(); self.ws1_material_delivered.set() 
            ws1_proc_thread = threading.Thread(target=self._process_at_workstation, args=('ws1', current_token, [0], self.ws1_output_buffer, self.ws1_processed))
            ws1_proc_thread.start()
            agv5_thread.join(); self.ws3_stock_material_delivered.set() 
            self.ws1_processed.wait(); self.ws1_processed.clear()
            if self.ws1_output_buffer and self.ws1_output_buffer[0].order_id == current_token.order_id: self.ws1_output_buffer.popleft() 
            agv3_thread = threading.Thread(target=self.agv_transfer_station_to_station, 
                                           args=('agv_3', current_token, 'ws1', 1, 'ws3', 0, 'ws1', 1, "SemiFinished_A1"))
            agv3_thread.start()
            agv3_thread.join(); self.ws3_semifinished_material_delivered.set() 
            self.ws3_stock_material_delivered.wait(); self.ws3_semifinished_material_delivered.wait()
            self.ws3_stock_material_delivered.clear(); self.ws3_semifinished_material_delivered.clear()
            ws3_proc_thread = threading.Thread(target=self._process_at_workstation, args=('ws3', current_token, [0, 2], self.ws3_output_buffer, self.ws3_processed))
            ws3_proc_thread.start()
            ws3_proc_thread.join() 
            self.ws3_processed.set(); self.ws3_processed.clear() 
            self.line_A_token_active = None 
            if self.ws3_output_buffer and self.ws3_output_buffer[0].order_id == current_token.order_id: self.ws3_output_buffer.popleft()
            agv7_task_thread = threading.Thread(target=self.agv_pickup_and_deliver_to_dispatch, args=('agv_7', current_token, 'ws3', 1))
            agv7_task_thread.daemon = True 
            with self.background_threads_lock: self.active_background_tasks.append(agv7_task_thread)
            agv7_task_thread.start()
        except Exception as e: 
            self.get_logger().error(f"Exception in Line A (T:{current_token.order_id if current_token else 'N/A'}): {e}", exc_info=e)
            if hasattr(self, 'line_A_token_active') and self.line_A_token_active == current_token : self.line_A_token_active = None

    def run_production_line_B(self, token):
        current_token = token
        try:
            self.line_B_token_active = current_token; current_token.set_line('B')
            agv2_thread = threading.Thread(target=self.agv_warehouse_pickup_and_deliver, args=('agv_2', current_token, 'ws2', 0))
            agv6_thread = threading.Thread(target=self.agv_warehouse_pickup_and_deliver, args=('agv_6', current_token, 'ws4', 2))
            agv2_thread.start(); agv6_thread.start()
            agv2_thread.join(); self.ws2_material_delivered.set()
            ws2_proc_thread = threading.Thread(target=self._process_at_workstation, args=('ws2', current_token, [0], self.ws2_output_buffer, self.ws2_processed))
            ws2_proc_thread.start()
            agv6_thread.join(); self.ws4_stock_material_delivered.set()
            self.ws2_processed.wait(); self.ws2_processed.clear()
            if self.ws2_output_buffer and self.ws2_output_buffer[0].order_id == current_token.order_id: self.ws2_output_buffer.popleft()
            agv4_thread = threading.Thread(target=self.agv_transfer_station_to_station,
                                           args=('agv_4', current_token, 'ws2', 1, 'ws4', 0, 'ws2', 1, "SemiFinished_B1"))
            agv4_thread.start()
            agv4_thread.join(); self.ws4_semifinished_material_delivered.set()
            self.ws4_stock_material_delivered.wait(); self.ws4_semifinished_material_delivered.wait()
            self.ws4_stock_material_delivered.clear(); self.ws4_semifinished_material_delivered.clear()
            ws4_proc_thread = threading.Thread(target=self._process_at_workstation, args=('ws4', current_token, [0, 2], self.ws4_output_buffer, self.ws4_processed))
            ws4_proc_thread.start()
            ws4_proc_thread.join() 
            self.ws4_processed.set(); self.ws4_processed.clear()
            self.line_B_token_active = None
            if self.ws4_output_buffer and self.ws4_output_buffer[0].order_id == current_token.order_id: self.ws4_output_buffer.popleft()
            agv8_task_thread = threading.Thread(target=self.agv_pickup_and_deliver_to_dispatch, args=('agv_8', current_token, 'ws4', 1))
            agv8_task_thread.daemon = True
            with self.background_threads_lock: self.active_background_tasks.append(agv8_task_thread)
            agv8_task_thread.start()
        except Exception as e: 
            self.get_logger().error(f"Exception in Line B (T:{current_token.order_id if current_token else 'N/A'}): {e}", exc_info=e)
            if hasattr(self, 'line_B_token_active') and self.line_B_token_active == current_token: self.line_B_token_active = None
    
    def generate_production_order(self):
        order_id = f"Order_{self.next_order_id:03d}"
        self.next_order_id += 1
        token = ProductionToken(order_id, time.perf_counter(), self.get_logger())
        return token

    def scenario_execution_loop(self, num_orders=4):
        orders_to_process = [self.generate_production_order() for _ in range(num_orders)]
        order_idx = 0; active_main_line_threads_map = {}; num_tokens_goal = num_orders
        while self.run and (len(self.fp_delivered_tokens) < num_tokens_goal): 
            finished_main_line_threads = [t for t in active_main_line_threads_map if not t.is_alive()]
            for t in finished_main_line_threads:
                if t in active_main_line_threads_map: del active_main_line_threads_map[t]
            if order_idx < num_orders:
                token_to_dispatch = orders_to_process[order_idx]; assigned_this_cycle = False
                if self.next_line_to_assign == 'A' and self.line_A_token_active is None:
                    thread = threading.Thread(target=self.run_production_line_A, args=(token_to_dispatch,))
                    active_main_line_threads_map[thread] = token_to_dispatch.order_id ; thread.start()
                    self.next_line_to_assign = 'B'; order_idx +=1; assigned_this_cycle = True
                elif self.next_line_to_assign == 'B' and self.line_B_token_active is None:
                    thread = threading.Thread(target=self.run_production_line_B, args=(token_to_dispatch,))
                    active_main_line_threads_map[thread] = token_to_dispatch.order_id; thread.start()
                    self.next_line_to_assign = 'A'; order_idx += 1; assigned_this_cycle = True
                if not assigned_this_cycle and (self.line_A_token_active is not None and self.line_B_token_active is not None):
                    if order_idx < num_orders: time.sleep(0.1) 
            with self.background_threads_lock: self.active_background_tasks = [t for t in self.active_background_tasks if t.is_alive()]
            if not self.run and active_main_line_threads_map: self.get_logger().warn(f"Scenario stopping, {len(active_main_line_threads_map)} main line production threads still potentially active...")
            if order_idx >= num_orders and not active_main_line_threads_map and len(self.fp_delivered_tokens) < num_tokens_goal:
                if not self.active_background_tasks: self.get_logger().warn(f"All orders dispatched to lines, main line threads done, no background AGV tasks, but {num_tokens_goal - len(self.fp_delivered_tokens)} tokens not yet at final point. Potential issue or waiting for final dispatch AGVs.")
            time.sleep(0.05) 
            
        final_message = "SIMULATION COMPLETE: ALL PRODUCTION ORDERS HAVE BEEN DELIVERED TO THE FINAL POINT."
        if not self.run and len(self.fp_delivered_tokens) < num_tokens_goal : 
            final_message = "SIMULATION INTERRUPTED: ORDER PROCESSING STOPPED BEFORE ALL ORDERS WERE DELIVERED."
        self.get_logger().info(final_message) 
        
        start_wait_bg = time.perf_counter(); max_wait_bg_time = 65 
        if self.active_background_tasks: 
            while True:
                with self.background_threads_lock:
                    self.active_background_tasks = [t for t in self.active_background_tasks if t.is_alive()]
                    if not self.active_background_tasks: break
                if not self.run and (time.perf_counter() - start_wait_bg > 10): self.get_logger().warn("Stop signal received during background task wait. Proceeding with shutdown."); break 
                if time.perf_counter() - start_wait_bg > max_wait_bg_time:
                    self.get_logger().warn(f"{len(self.active_background_tasks)} background AGV tasks may not have completed after {max_wait_bg_time}s timeout.") 
                    break
                time.sleep(0.5)
        self.run = False

    def start_scenario(self, num_orders=10):
        if self.run: 
            self.get_logger().warn("Scenario is already running.")
            return 
        
        self.run = True
        self.simulation_start_time = time.perf_counter() 
        self.fp_delivered_tokens.clear()
        self.next_order_id = 1 
        
        for k in self.occupancy: self.occupancy[k] = [False] * len(self.occupancy[k])
        for b in [self.ws1_output_buffer, self.ws2_output_buffer, self.ws3_output_buffer, self.ws4_output_buffer]: b.clear()
        for ev_event in [self.ws1_material_delivered, self.ws1_processed, self.ws2_material_delivered, self.ws2_processed,
                         self.ws3_stock_material_delivered, self.ws3_semifinished_material_delivered, self.ws3_processed,
                         self.ws4_stock_material_delivered, self.ws4_semifinished_material_delivered, self.ws4_processed]: 
            ev_event.clear()
        
        with self.background_threads_lock: self.active_background_tasks = [] 
        self.llm_conversation_history = [{"role": "system", "content": self.llm_system_prompt}]
        self.current_target_opm = 0.5 
        self.operator_instructions = "" 

        self.agv_target_speeds = {agv_id: 1.0 for agv_id in self.agv_ids_list}

        self.get_logger().info("Setting initial default target velocities for all AGVs...")
        for agv_id, default_speed in self.agv_target_speeds.items():
            agv_instance = self.get_agv_instance(agv_id)
            if agv_instance:
                agv_instance.target_velocity = default_speed
            else:
                self.get_logger().warn(f"Could not set initial speed for AGV {agv_id}: instance not found.")

        priming_thread = threading.Thread(target=self._perform_llm_priming_in_background)
        priming_thread.daemon = True 
        with self.background_threads_lock: self.active_background_tasks.append(priming_thread)
        priming_thread.start()
        
        self.main_scenario_thread_ref = threading.Thread(target=self.scenario_execution_loop, args=(num_orders,))
        self.main_scenario_thread_ref.daemon = False 
        self.main_scenario_thread_ref.start()

    def stop_scenario(self):
        self.get_logger().warn("Stopping scenario (setting self.run = False)...") 
        self.run = False 
        if hasattr(self, 'main_scenario_thread_ref') and self.main_scenario_thread_ref.is_alive():
            self.main_scenario_thread_ref.join(timeout=15) 
            if self.main_scenario_thread_ref.is_alive():
                self.get_logger().warn("Main scenario loop did not complete within the stop timeout.") 
        self.get_logger().info("Scenario stop procedure initiated.")