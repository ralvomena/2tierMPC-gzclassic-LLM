import tkinter as tk
from tkinter import ttk

gui_win = None

class NetemGui:
    """
    Class that builds the Netem GUI with Tkinter.
    """
    def __init__(self, master, node_instance):
        master.title('Netem GUI')
        master.columnconfigure(0, weight=1)
        self.node = node_instance

        self.node.get_logger().info("Netem GUI Start.")
        
        main = ttk.Frame(master, padding=(30,10))
        main.grid()

        delay = tk.StringVar()
        distribution = tk.StringVar()
        loss = tk.StringVar()
        timer = tk.StringVar()

        top_label = ttk.Label(main, text="Apply to all")
        delay_label = ttk.Label(main, text="Delay (ms)")
        delay_entry = ttk.Entry(main, width=2, textvariable=delay)
        dist_label = ttk.Label(main, text="Distribution (ms)")
        dist_entry = ttk.Entry(main, width=2, textvariable=distribution)
        loss_label = ttk.Label(main, text="Loss (%)")
        loss_entry = ttk.Entry(main, width=2, textvariable=loss)
        timer_label = ttk.Label(main, text="Timer (s)")
        timer_entry = ttk.Entry(main, width=2, textvariable=timer)
        apply_button = ttk.Button(main, text="Apply",
                        command=lambda:self.node.apply_netem_all('enable', delay.get(), distribution.get(), loss.get()))
        apply_timer_button = ttk.Button(main, text="Apply with timer", 
                        command=lambda: self.node.apply_netem_all('enable', delay.get(), distribution.get(), loss.get(), timer.get()))
        remove_button = ttk.Button(main, text="Remove",
                        command=lambda:self.node.apply_netem_all('disable'))

        top_label.grid(column=0, row=0, columnspan=2, padx=5, pady=5)
        delay_label.grid(column=0, row=1, sticky="W", padx=5, pady=5)
        delay_entry.grid(column=1, row=1, sticky="EW", padx=5, pady=5)
        dist_label.grid(column=0, row=2, sticky="W", padx=5, pady=5)
        dist_entry.grid(column=1, row=2, sticky="EW", padx=5, pady=5)
        loss_label.grid(column=0, row=3, sticky="W", padx=5, pady=5)
        loss_entry.grid(column=1, row=3, sticky="EW", padx=5, pady=5)
        timer_label.grid(column=0, row=4, sticky="W", padx=5, pady=5)
        timer_entry.grid(column=1, row=4, sticky="EW", padx=5, pady=5)
        apply_button.grid(column=0, row=5, sticky="EW", padx=5, pady=5)
        apply_timer_button.grid(column=1, row=5, padx=5, pady=5)
        remove_button.grid(columnspan=2, sticky="EW", row=6, padx=5, pady=5)
        delay_entry.focus()
        
        self.srv_frame = ttk.Frame(master, padding=(30,10))
        self.srv_frame.grid()
        self.srv_row = 1

    # def add_srv_entry(self, agv):
    #     delay = tk.StringVar()
    #     distribution = tk.StringVar()
    #     loss = tk.StringVar()
    #     timer = tk.StringVar()

    #     agv_label = ttk.Label(self.srv_frame, text=agv.upper() + ':')
    #     delay_label = ttk.Label(self.srv_frame, text="Delay (ms)")
    #     delay_entry = ttk.Entry(self.srv_frame, width=5, textvariable=delay)
    #     dist_label = ttk.Label(self.srv_frame, text="Dist. (ms)")
    #     dist_entry = ttk.Entry(self.srv_frame, width=5, textvariable=distribution)
    #     loss_label = ttk.Label(self.srv_frame, text="Loss (%)")
    #     loss_entry = ttk.Entry(self.srv_frame, width=5, textvariable=loss)
    #     timer_label = ttk.Label(self.srv_frame, text="Timer (s)")
    #     timer_entry = ttk.Entry(self.srv_frame, width=5, textvariable=timer)
    #     apply_button = ttk.Button(self.srv_frame, text="Apply",
    #                     command=lambda:self.node.apply_netem_individually(agv, 'enable', delay.get(), distribution.get(), loss.get()))
    #     apply_timer_button = ttk.Button(self.srv_frame, text="Apply with timer", 
    #                     command=lambda:self.node.apply_netem_individually(agv, 'enable', delay.get(), distribution.get(), loss.get(), timer.get()))
    #     remove_button = ttk.Button(self.srv_frame, text="Remove",
    #                     command=lambda:self.node.apply_netem_individually(agv, 'disable'))

    #     agv_label.grid(column=0, row=self.srv_row, sticky="W", padx=5, pady=5)
    #     delay_label.grid(column=1, row=self.srv_row, sticky="W", padx=5, pady=5)
    #     delay_entry.grid(column=2, row=self.srv_row, sticky="EW", padx=5, pady=5)
    #     dist_label.grid(column=3, row=self.srv_row, sticky="W", padx=5, pady=5)
    #     dist_entry.grid(column=4, row=self.srv_row, sticky="EW", padx=5, pady=5)
    #     loss_label.grid(column=5, row=self.srv_row, sticky="W", padx=5, pady=5)
    #     loss_entry.grid(column=6, row=self.srv_row, sticky="EW", padx=5, pady=5)
    #     timer_label.grid(column=7, row=self.srv_row, sticky="W", padx=5, pady=5)
    #     timer_entry.grid(column=8, row=self.srv_row, sticky="EW", padx=5, pady=5)
    #     apply_button.grid(column=9, row=self.srv_row, sticky="EW",padx=5, pady=5)
    #     apply_timer_button.grid(column=10, row=self.srv_row, sticky="EW",padx=5, pady=5)
    #     remove_button.grid(column=11, row=self.srv_row, sticky="EW",padx=5, pady=5)

    #     self.srv_row += 1
