import tkinter as tk
from tkinter import ttk
from datetime import datetime

class SupervisorGui:
    """
    Class that builds the Supervisor GUI with Tkinter.
    """

    def __init__(self, master, node_instance):
        self.master = master
        master.title('Supervisor GUI')
        master.geometry("850x600")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)

        self.node = node_instance

        self.agv_list = []
        self.tree_inserts = {}

        # State variables for Toplevel windows
        self.mpc_window_instance = None
        self.llm_window_instance = None
        
        # Row counter for the MPC window
        self.mpc_row = 1
        
        self.main = ttk.Frame(master, padding="10 10 10 10")
        self.main.grid(sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main.columnconfigure(0, weight=1)

        # Widget variables
        self.info_text = None
        self.agv_tree = None
        self.llm_input_text = None

        self._create_gui_elements()
        
        self._log_info("Information panel initialized. GUI is ready.")

    def _create_gui_elements(self):
        """Creates and lays out the main GUI elements."""
        current_row = 0

        self._create_treeview_panel(self.main, current_row)
        current_row += 2 

        self._create_control_buttons_panel(self.main, current_row)
        current_row += 2

        self._create_info_panel(self.main, current_row)
        
        # Configure row weights after all widgets are created to ensure proper resizing.
        self.main.rowconfigure(self.agv_tree.grid_info()['row'], weight=3)
        self.main.rowconfigure(self.info_text.grid_info()['row'], weight=2)

    def _create_treeview_panel(self, parent, start_row):
        top_label = ttk.Label(parent, text="AGVs in Gazebo", font=("Arial", 12, "bold"))
        top_label.grid(column=0, row=start_row, columnspan=5, padx=5, pady=(5,0), sticky=tk.W)

        columns = ('agv_id', 'pose', 'velocity', 'target_speed', 'mpc_edge', 'mpc_local')
        self.agv_tree = ttk.Treeview(parent, columns=columns, show='headings', height=7)

        self.agv_tree.column('agv_id', anchor="center", stretch="no", width=80)
        self.agv_tree.column('pose', anchor="w", stretch="yes", width=220)
        self.agv_tree.column('velocity', anchor="w", stretch="yes", width=200)
        self.agv_tree.column('target_speed', anchor="center", stretch="no", width=150)
        self.agv_tree.column('mpc_edge', anchor="center", stretch="no", width=100)
        self.agv_tree.column('mpc_local', anchor="center", stretch="no", width=100)

        self.agv_tree.heading('agv_id', text='AGV ID')
        self.agv_tree.heading('pose', text='Pose [x(m), y(m), \u03B8(\u00B0)]')
        self.agv_tree.heading('velocity', text='Velocity [v(m/s), \u03C9(rad/s)]')
        self.agv_tree.heading('target_speed', text='Target speed (m/s)')
        self.agv_tree.heading('mpc_edge', text='Edge MPC')
        self.agv_tree.heading('mpc_local', text='Local MPC')

        self.agv_tree.grid(row=start_row + 1, column=0, columnspan=5, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        scroll_agv_tree = ttk.Scrollbar(parent, orient='vertical', command=self.agv_tree.yview)
        scroll_agv_tree.grid(column=5, row=start_row + 1, sticky='ns', pady=5)
        self.agv_tree.configure(yscrollcommand=scroll_agv_tree.set)

    def _create_control_buttons_panel(self, parent, start_row):
        separator = ttk.Separator(parent, orient='horizontal')
        separator.grid(column=0, row=start_row, columnspan=6, sticky=tk.EW, padx=5, pady=5)
        buttons_frame = ttk.Frame(parent)
        buttons_frame.grid(column=0, row=start_row + 1, columnspan=6, sticky=tk.EW, padx=5, pady=5)

        mpc_button = ttk.Button(buttons_frame, text="MPC Service", command=self.open_mpc_window)
        mpc_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        scn_button = ttk.Button(buttons_frame, text="Start Scenario", command=self.node.start_scenario)
        scn_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        stop_scn_button = ttk.Button(buttons_frame, text="Stop Scenario", command=self.node.stop_scenario)
        stop_scn_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        llm_button = ttk.Button(buttons_frame, text="Open LLM Panel", command=self.open_llm_window)
        llm_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

    def _create_info_panel(self, parent, start_row):
        separator = ttk.Separator(parent, orient='horizontal')
        separator.grid(column=0, row=start_row, columnspan=6, sticky=tk.EW, padx=5, pady=5)
        info_label = ttk.Label(parent, text="Information Panel", font=("Arial", 10, "bold"))
        info_label.grid(column=0, row=start_row + 1, columnspan=5, sticky=tk.W, padx=5, pady=(5,0))
        
        self.info_text = tk.Text(parent, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.info_text.grid(column=0, columnspan=5, row=start_row + 2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        scroll_info = ttk.Scrollbar(parent, orient='vertical', command=self.info_text.yview)
        scroll_info.grid(column=5, row=start_row + 2, sticky='ns', pady=5)
        self.info_text.configure(yscrollcommand=scroll_info.set)

    def _log_info(self, message):
        """Adds a timestamped message to the information panel."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        full_message = f"[{timestamp}] {message}\n"
        
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, full_message)
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)
    
    def _create_llm_panel(self, parent_window):
        """Creates the LLM GUI elements (input and send button only)."""
        parent_window.columnconfigure(0, weight=1)
        
        input_frame = ttk.LabelFrame(parent_window, text="Your Query", padding=5)
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=(10,5))
        input_frame.columnconfigure(0, weight=1); input_frame.rowconfigure(0, weight=1)
        
        self.llm_input_text = tk.Text(input_frame, height=5, wrap=tk.WORD)
        self.llm_input_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        llm_input_scroll = ttk.Scrollbar(input_frame, orient='vertical', command=self.llm_input_text.yview)
        llm_input_scroll.grid(row=0, column=1, sticky='ns')
        self.llm_input_text.configure(yscrollcommand=llm_input_scroll.set)
        
        send_llm_button = ttk.Button(parent_window, text="Send to LLM", command=self._communicate_with_llm)
        send_llm_button.grid(row=1, column=0, sticky=tk.EW, padx=10, pady=(5,10))
        
        parent_window.rowconfigure(0, weight=1)

    def _destroy_llm_window(self):
        if self.llm_window_instance and self.llm_window_instance.winfo_exists():
            self.llm_window_instance.destroy()
        self.llm_window_instance = None

    def open_llm_window(self):
        """Opens the LLM panel window, or brings it to the front if it already exists."""
        if self.llm_window_instance and self.llm_window_instance.winfo_exists():
            self.llm_window_instance.lift()
            return
        self.llm_window_instance = tk.Toplevel(self.master)
        self.llm_window_instance.title("LLM Panel")
        self.llm_window_instance.geometry("500x200")
        self.llm_window_instance.protocol("WM_DELETE_WINDOW", self._destroy_llm_window)
        self.llm_window_instance.transient(self.master)
        self._create_llm_panel(self.llm_window_instance)
        self.llm_input_text.focus_set()

    def _communicate_with_llm(self):
        user_query = self.llm_input_text.get("1.0", tk.END).strip()
        if not user_query:
            self._log_info("LLM query is empty.")
            return

        self._log_info(f"Sending to LLM: '{user_query}'")
        if hasattr(self.node, 'send_operator_command_to_llm'):
            self.node.send_operator_command_to_llm(user_query)
        
        # Optional: Clear the input field after sending
        self.llm_input_text.delete("1.0", tk.END)

    def insert_agv_to_tree(self, agv, pose, velocity, target_speed, mpc_edge_status, mpc_local_status):
        x = f"{pose[0]:.2f}"; y = f"{pose[1]:.2f}"; theta = f"{pose[2]:.2f}"
        linear = f"{velocity[0]:.2f}"; angular = f"{velocity[1]:.2f}"
        values = (agv.upper(), f'[{x}, {y}, {theta}]', f'[{linear}, {angular}]', target_speed, mpc_edge_status, mpc_local_status)
        if agv in self.tree_inserts:
            self.agv_tree.item(self.tree_inserts[agv], values=values)
        else:
            insert_id = self.agv_tree.insert('', 'end', values=values)
            self.tree_inserts[agv] = insert_id
            if agv not in self.agv_list: self.agv_list.append(agv)
    
    def remove_agv_from_tree(self, agv_id_to_remove):
        if agv_id_to_remove in self.tree_inserts:
            item_id = self.tree_inserts.pop(agv_id_to_remove)
            self.agv_tree.delete(item_id)
            if agv_id_to_remove in self.agv_list: self.agv_list.remove(agv_id_to_remove)
            self._log_info(f"AGV {agv_id_to_remove.upper()} removed from GUI.")
            if self.mpc_window_instance and self.mpc_window_instance.winfo_exists():
                self.open_mpc_window(force_recreate=True)
    
    def _destroy_mpc_window(self):
        if self.mpc_window_instance and self.mpc_window_instance.winfo_exists(): self.mpc_window_instance.destroy()
        self.mpc_window_instance = None

    def open_mpc_window(self, force_recreate=False):
        if not force_recreate and self.mpc_window_instance and self.mpc_window_instance.winfo_exists(): self.mpc_window_instance.lift(); return
        if self.mpc_window_instance and self.mpc_window_instance.winfo_exists(): self.mpc_window_instance.destroy()
        
        self.mpc_window_instance = tk.Toplevel(self.master)
        self.mpc_window_instance.title("MPC Service")
        self.mpc_window_instance.protocol("WM_DELETE_WINDOW", self._destroy_mpc_window)
        self.mpc_window_instance.transient(self.master)
        self.mpc_window_instance.grab_set()
        self.mpc_row = 0
        
        # Create a scrollable frame to hold the AGV entries
        canvas = tk.Canvas(self.mpc_window_instance)
        scrollbar = ttk.Scrollbar(self.mpc_window_instance, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        if not self.agv_list: 
            ttk.Label(scrollable_frame, text="No AGVs available.").grid(row=0, column=0, padx=10, pady=10)
        else:
            for agv in self.agv_list: 
                self._add_mpc_entry_to_window(scrollable_frame, agv)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _add_mpc_entry_to_window(self, parent_frame, agv):
        agv_frame = ttk.Frame(parent_frame, padding=5)
        agv_frame.grid(row=self.mpc_row, column=0, sticky=tk.EW, pady=2)
        ttk.Label(agv_frame, text=agv.upper() + ':', width=10).grid(row=0, column=0, sticky="W")
        ttk.Label(agv_frame, text="x(m)").grid(row=0, column=1, sticky="W", padx=(5,0))
        x_entry = ttk.Entry(agv_frame, width=5)
        x_entry.grid(row=0, column=2, padx=(0,5))
        ttk.Label(agv_frame, text="y(m)").grid(row=0, column=3, sticky="W", padx=(5,0))
        y_entry = ttk.Entry(agv_frame, width=5)
        y_entry.grid(row=0, column=4, padx=(0,5))
        ttk.Label(agv_frame, text='\u03B8(\u00B0)').grid(row=0, column=5, sticky="W", padx=(5,0))
        theta_entry = ttk.Entry(agv_frame, width=5)
        theta_entry.grid(row=0, column=6, padx=(0,10))
        call_cmd = lambda a=agv, x=x_entry, y=y_entry, th=theta_entry: self.mpc_service_call(a, 'move', x.get(), y.get(), th.get())
        ttk.Button(agv_frame, text="Call", command=call_cmd).grid(row=0, column=7, padx=2)
        stop_cmd = lambda a=agv: self.stop_mpc(a)
        ttk.Button(agv_frame, text="Stop", command=stop_cmd).grid(row=0, column=8, padx=2)
        self.mpc_row += 1
    
    def mpc_service_call(self, agv, action, x_str, y_str, theta_str):
        try: 
            x_val = float(x_str)
            y_val = float(y_str)
            theta_val = float(theta_str)
        except ValueError:
            self._log_info(f"Error: Invalid input for {agv.upper()}. Use numbers for coordinates.")
            if self.mpc_window_instance and self.mpc_window_instance.winfo_exists():
                self.mpc_window_instance.lift()
            return
        for agv_instance in self.node.agv_instances:
            if agv_instance.agv_id == agv:
                agv_instance.call_mpc(action, x_val, y_val, theta_val)
                self._log_info(f"Command '{action}' sent to {agv.upper()} with x:{x_val}, y:{y_val}, \u03B8:{theta_val}.")
                return
        self._log_info(f"AGV {agv.upper()} not found for MPC call.")

    def stop_mpc(self, agv):
        for agv_instance in self.node.agv_instances:
            if agv_instance.agv_id == agv:
                agv_instance.call_mpc('stop', 0.0, 0.0, 0.0)
                self._log_info(f"Stopping {agv.upper()}.")
                return
        self._log_info(f"AGV {agv.upper()} not found for MPC stop.")
