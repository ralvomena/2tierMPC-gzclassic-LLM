import casadi as cas
import numpy as np
import time
import threading

# Thread lock
lock = threading.Lock()

class MPC:
    """
    MPC class.
    """

    def __init__(self, N, T, Q, R, ma57_solver=True):
        """
        MPC core class for tracking.
        Cost function building and solution extraction are implemented here.
        """

        self.x0 = None # AGV initial state and current state during simulation
        self.xs = None # AGV reference position
        self.N = N # Prediction horizon
        self.T = T # MPC time step
        self.Q = Q # Q weight matrix (state)
        self.R = R # R weight matrix (control signals)
        self.ma57 = ma57_solver # Boolean of MA57 solver (make it False at the MPCLocalNode super() if the MA57 is not installed)
        self.trajectory_update = False # Boolean of trajectory update
        self.trajectory = [] # Planned trajectory
        self.mpc_start = 0 # Time at the start of MPC iteration
        self.mpc_iter = 0 # Iterations counter 
        self.mpc_time = 0 # Time interval of a MPC iteration
        self.mpc_time_avg = 0 # Time interval average of MPC iterations
        self.mpc_run = False # Boolean of MPC running

        # This statement will enable MA57 on IPOPT options if self.ma57 variable is True
        if self.ma57:
            self.opts = {'ipopt.linear_solver': 'ma57', 'ipopt.max_iter': 2000, 'ipopt.print_level': 0, 'print_time': 0,
                         'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}
        else:
            self.opts = {'ipopt.max_iter': 2000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                         'ipopt.acceptable_obj_change_tol': 1e-6}

    def cost_function(self):
        """
        Function to build the MPC cost function.
        """

        # State variables
        self.x = cas.SX.sym('x')
        self.y = cas.SX.sym('y')
        self.theta = cas.SX.sym('theta')
        self.states = cas.vertcat(self.x, self.y, self.theta)
        self.n_states = self.states.shape

        # Control variables
        self.v = cas.SX.sym('v')
        self.omega = cas.SX.sym('omega')
        self.controls = cas.vertcat(self.v, self.omega)
        self.n_controls = self.controls.shape

        # Set initial/default values for ​runtime speed limits
        self.runtime_v_max = 1.0
        self.runtime_v_min = -1.0

        # Right side of differential drive robot model equation
        self.rhs = cas.vertcat(self.v * cas.cos(self.theta), self.v * cas.sin(self.theta), self.omega)

        # System model
        self.f = cas.Function('f', [self.states, self.controls], [self.rhs])

        # Control optimization variables
        self.U = cas.SX.sym('U', self.n_controls[0], self.N)

        # State optimization variables
        self.X = cas.SX.sym('X', self.n_states[0], (self.N + 1))

        # Parameters: initial or current state and references
        self.P = cas.SX.sym('P', self.n_states[0] + self.N * (self.n_states[0] + self.n_controls[0]))

        # Control optimization variables
        self.u0 =  np.zeros((self.N, 2))

        # State optimization variables (the initial state is replicated to be used as an initial guess)
        self.X0 = cas.repmat(self.x0, 1, self.N + 1).T

        # Constraints vector
        self.g = []

        # Cost funtion
        self.cost = 0
        self.st = self.X[:, 0]
        self.g = cas.vertcat(self.g, self.st - self.P[0:3]) # Multiple shooting constraints on initial state
        for k in range(0, self.N):
            self.st = self.X[:, k]
            self.con = self.U[:, k]
            self.cost = self.cost + (self.st-self.P[5*k+3:5*k+6]).T@self.Q@(self.st-self.P[5*k+3:5*k+6]) + \
                       (self.con-self.P[5*k+6:5*k+8]).T@self.R@(self.con-self.P[5*k+6:5*k+8]) # Calculate cost function
            self.st_next = self.X[:, k+1]
            k1 = self.f(self.st, self.con)
            k2 = self.f(self.st + self.T/2*k1, self.con)
            k3 = self.f(self.st + self.T/2*k2, self.con)
            k4 = self.f(self.st + self.T*k3, self.con)
            self.st_next_RK4 = self.st + (k1 + 2*k2 + 2*k3 + k4)*self.T/6 # Next state with RK4 
            self.g = cas.vertcat(self.g, self.st_next - self.st_next_RK4) # Multiple shooting constraints

        # Optimization variables vector
        self.opt_variables = cas.vertcat(cas.reshape(self.X, 3 * (self.N+1), 1), cas.reshape(self.U, 2 * self.N, 1))

        # NLP problem
        self.nlp_prob = {'f': self.cost, 'x': self.opt_variables, 'g': self.g, 'p': self.P}

        # Solver
        self.solver = cas.nlpsol('solver', 'ipopt', self.nlp_prob, self.opts)

        # Equality constraints value vectors
        self.lbg_values = np.zeros(3*(self.N+1))
        self.ubg_values = np.zeros(3*(self.N+1))

        # Inequality constraints value vectors
        self.lbx_values = np.zeros((3*(self.N+1)+2*self.N, 1))
        self.ubx_values = np.zeros((3*(self.N+1)+2*self.N, 1))

        # Indexes list for states and controls
        self.x_index = np.arange(0, 3*(self.N+1), 3).tolist()
        self.y_index = np.arange(1, 3*(self.N+1), 3).tolist()
        self.theta_index = np.arange(2, 3*(self.N+1), 3).tolist()
        self.v_index = np.arange(3*(self.N+1), 3*(self.N+1) + 2 * self.N, 2).tolist()
        self.omega_index = np.arange(3*(self.N+1)+1, 3*(self.N+1) + 2*self.N, 2).tolist()

        # Bounds of state constraints (limited to scenario size)
        self.lbx_values[self.x_index, :] = self.scn_size[0]
        self.lbx_values[self.y_index, :] = self.scn_size[1]
        self.lbx_values[self.theta_index, :] = -float("inf")
        self.ubx_values[self.x_index, :] = self.scn_size[2]
        self.ubx_values[self.y_index, :] = self.scn_size[3]
        self.ubx_values[self.theta_index, :] = float("inf")
        
        # Set angular velocity
        self.omega_max = cas.pi/4
        self.omega_min = -self.omega_max

        # Bounds of control constraints
        self.lbx_values[self.v_index, :] = self.runtime_v_min
        self.lbx_values[self.omega_index, :] = self.omega_min
        self.ubx_values[self.v_index, :] = self.runtime_v_max
        self.ubx_values[self.omega_index, :] = self.omega_max

        # Arguments dictionary
        self.args = {'lbg': self.lbg_values, 'ubg': self.ubg_values, 
                     'lbx': self.lbx_values, 'ubx': self.ubx_values}

        # Show terminal info
        self.get_logger().info('Cost function has been successfully built.')

    def set_xs(self, x, y, theta):
        """
        Function to set the AGV reference.

        :param x: x reference
        :param y: y reference
        :param theta: theta reference
        """
        self.xs = cas.DM([x, y, theta])

    def set_linear_velocity(self, v):
        """
        Function to set the AGV linear velocity.

        :param v: linear velocity
        """
        self.runtime_v_max = v
        self.runtime_v_min = -v
        self.args['lbx'][self.v_index, :] = self.runtime_v_min
        self.args['ubx'][self.v_index, :] = self.runtime_v_max

    def start_mpc(self):
        """
        This function updates the values of p and x0 keys from the arguments dictionary (self.args) and then calls the MPC solver. 
        p is updated with the trajectory received from the edge. If a new trajectory has not been received at the next call of this function 
        (managed by a ROS 2 timer), the second element of the trajectory vector is brought to the first position, and the last is repeated to 
        preserve the vector size. x0 is updated with the results of the last iteration. It works as an initial guess of the next solution and helps the 
        solver to accelerate convergence. After this, the function finally calls the solver.
        The timer which calls this function is canceled when the AGV pose minus the reference is below a tolerance.
        """

        self.error = np.linalg.norm(self.x0 - self.xs)
        if self.error < self.tolerance:
            self.mpc_run = False
            self.mpc_timer.cancel()
            self.call_velocity_publisher(0.0, 0.0)
            self.call_close_mpc()
            self.get_logger().info(f'O {self.agv_id.upper()} reached the reference. Closing MPC...')
            self.get_logger().info(f'MPC iterations time interval average: {round(self.mpc_time_avg*1000, 2)} ms')
            return
        
        self.mpc_start = time.perf_counter()
        self.args.update({'p': cas.horzcat(self.x0[0], self.x0[1], self.x0[2])})
        
        if self.trajectory_update:
            lock.acquire()
            self.trajectory_update = False
            lock.release()
            self.mpc_run = True
            for k in range(0, self.N):
                x_ref = self.trajectory[k, 0]
                y_ref = self.trajectory[k, 1]
                theta_ref = self.trajectory[k, 2]
                u_ref = 0
                omega_ref = 0
                self.args.update({'p': cas.horzcat(self.args['p'], x_ref, y_ref, theta_ref, u_ref, omega_ref)})
                self.args.update({'x0': cas.vertcat(cas.reshape(self.X0.T, 3 * (self.N + 1), 1), cas.reshape(self.u0.T, 2 * self.N, 1))})
            self.solve_mpc()
        else:
            if self.mpc_run:
                self.trajectory_roll = np.roll(self.trajectory, -1, axis=0)
                self.trajectory_roll[-1] = self.trajectory_roll[self.trajectory_roll.shape[0] - 2]
                for k in range(0, self.N):
                    x_ref = self.trajectory_roll[k, 0]
                    y_ref = self.trajectory_roll[k, 1]
                    theta_ref = self.trajectory_roll[k, 2]
                    u_ref = 0
                    omega_ref = 0
                    self.args.update({'p': cas.horzcat(self.args['p'], x_ref, y_ref, theta_ref, u_ref, omega_ref)})
                    self.args.update({'x0': cas.vertcat(cas.reshape(self.X0.T, 3 * (self.N + 1), 1), cas.reshape(self.u0.T, 2 * self.N, 1))})
                self.solve_mpc()


    def solve_mpc(self):
        """
        This function calls the solver to obtain the MPC solution with the trajectory and control signals.
        """
        sol = self.solver(x0=self.args['x0'], lbx=self.args['lbx'], ubx=self.args['ubx'], lbg=self.args['lbg'],
                          ubg=self.args['ubg'], p=self.args['p'])
        self.mpc_end = time.perf_counter()
        t1 = threading.Thread(target=self.get_x, args=sol['x'].full().T)
        t2 = threading.Thread(target=self.get_u, args=sol['x'].full().T)
        t1.start()
        t2.start()
        self.mpc_time = self.mpc_time + (self.mpc_end - self.mpc_start)
        self.mpc_iter = self.mpc_iter + 1
        self.mpc_time_avg = self.mpc_time / self.mpc_iter
        if self.save_sim_data:
            self.mpc_time_file.write(f'{self.mpc_end - self.mpc_start}\n')
        t1.join()
        t2.join()

    def get_x(self, sol):
        """
        This function extracts x (predicted states) from the solution to update the self.X0 vector. 
        It also eliminates the vector's first element and repeats the last one. This variable is used as an 
        initial guess for the next MPC iteration.

        :param sol: MPC solution
        """
        self.X0 = sol[0:3 * (self.N + 1)]
        self.X0 = cas.reshape(self.X0, 3, self.N + 1).T
        self.X0 = cas.vertcat(self.X0[1:, :], self.X0[self.X0.shape[0] - 1, :])

    def get_u(self, sol):
        """
        This function extracts u (control signals) from the solution to update the self.u0 vector. 
        It also eliminates the vector's first element and repeats the last one. This variable is used as an
        initial guess for the next MPC iteration. The function also calls the call_velocity_publisher function 
        to publish the first element of the control signal vector, and the save_control_signal_data function to
        save the published control signals if the data saving is enabled.

        :param sol: MPC solution
        """
        u = sol[3 * (self.N + 1):]
        u = cas.reshape(u, 2, self.N).T
        t = threading.Thread(target=self.call_velocity_publisher, args=(float(u[0, 0]), float(u[0, 1])))
        t.start()
        self.u0 = cas.vertcat(u[1:, :], u[u.shape[0] - 1, :])
        if self.save_sim_data and self.mpc_run:
            threading.Thread(target=self.save_control_signal, args=(u[0, 0], u[0, 1])).start()
