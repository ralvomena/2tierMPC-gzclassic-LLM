import time
import numpy as np
import threading
import casadi as cas
import edge_tier_pkg.obstacles as obs

# Messages interfaces
from interfaces_pkg.msg import Pose, Trajectory

class MPC:
    """
    MPC core class from edge.
    Cost function building, collision avoidance procedures and solution extraction are implemented here.
    """

    def __init__(self, N, T, Q, R, ma57_solver=False):
        """
        MPC constructor.
        """

        self.n = 1 # Number of AVGs in scenario. This variable is updated in mpc node.
        self.x0 = [0.0, 0.0, 0.0] # AGV initial state and current state during simulation
        self.xs = None # AGV reference position
        self.N = N # Prediction horizon
        self.T = T # MPC time step
        self.Q = Q # Q weight matrix (state)
        self.R = R # R weight matrix (control signals)
        self.error = 0 # State error (state and reference distance)
        self.x0_update = False # State update flag
        self.receive_iter = 0 # Count received states
        self.sol_x = cas.DM(1, 2*N) # Store the MPC solution (states)
        self.sol_u = cas.DM(1, 3*(N+1)) # Store the MPC solution (control)
        self.ma57 = ma57_solver # MA57 preference (enable or disable)
        self.mpc_start = 0 # MPC start time
        self.mpc_iter = 0 # MPC iterations counter
        self.mpc_time = 0 # Time interval of an MPC iteration
        self.mpc_time_avg = 0 # Time interval average
        self.mpc_run = False # MPC run flag
        self.trajectory = np.array([]) # Store the generated trajectory
        self.pkg_id = 0 # Package identifier

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

        # Parameters: initial or current state and references. Parameters size is adjusted according to quantity of AGVs in scenario.
        if self.limit_n > 0:
            self.P = cas.SX.sym('P', (2 * self.n_states[0] + (self.limit_n)*2*(self.N+1) + 3*self.obstacles))
        else:
            self.P = cas.SX.sym('P', (2 * self.n_states[0] + (self.n-1)*2*(self.N+1) + 3*self.obstacles))

        # Control optimization variables
        self.u0 = np.zeros((self.N, self.n_controls[0]))

        # State optimization variables (the initial state is replicated to be used as an initial guess)
        self.X0 = cas.repmat(self.x0, 1, self.N + 1).T

        # Constraints vector
        self.g = []

        # Cost function
        self.obj = 0
        self.st = self.X[:, 0]
        self.g = cas.vertcat(self.g, self.st - self.P[0:3])
        for k in range(0, self.N):
            self.st = self.X[:, k]
            self.con = self.U[:, k]
            self.obj = self.obj + (self.st-self.P[3:6]).T@self.Q@(self.st-self.P[3:6]) + self.con.T@self.R@self.con
            self.st_next = self.X[:, k + 1]
            k1 = self.f(self.st, self.con)
            k2 = self.f(self.st + self.T/2*k1, self.con)
            k3 = self.f(self.st + self.T/2*k2, self.con)
            k4 = self.f(self.st + self.T*k3, self.con)
            self.st_next_RK4 = self.st + (k1 + 2*k2 + 2*k3 + k4)*self.T/6
            self.g = cas.vertcat(self.g, self.st_next - self.st_next_RK4)

         # Optimization variables vector
        self.opt_variables = cas.vertcat(cas.reshape(self.X, 3*(self.N+1), 1), cas.reshape(self.U, 2*self.N, 1))

        # Collision avoidance with other AGVS
        if self.n > 1:
            if self.limit_n > 0:
                self.col_avoidance_parameters = cas.DM((self.limit_n)*2*(self.N+1), 1)
            else:
                self.col_avoidance_parameters = cas.DM((self.n-1)*2*(self.N+1), 1)
            self.col_avoidance_parameters[:, :] = -float('inf')

            for i in range(0, (self.n-1)*2*(self.N+1), 2*(self.N+1)):
                k = 0
                for j in range(self.N+1):
                    self.g = cas.vertcat(self.g, -cas.sqrt((self.X[0, j] - self.P[6+j+k+i]) ** 2 +
                                        (self.X[1, j] - self.P[6+j+k+i+1]) ** 2) +
                                                    self.agv_diam + self.d_safe)
                    k = k + 1

        #  Collision avoidance with fixed obstacles
        if self.obstacles > 0:
            for j in range(self.N+1):
                for k in range(0, 3*self.obstacles, 3):
                    self.g = cas.vertcat(self.g, -cas.sqrt((self.X[0, j] - self.P[6+(self.n-1)*2*(self.N+1)+k])**2 
                                + (self.X[1, j] - self.P[6+(self.n-1)*2*(self.N+1)+k+1])**2) + self.agv_diam/2 + 
                                self.P[6+(self.n-1)*2*(self.N+1)+k+2]/2 + self.d_safe_obs)

        # NLP problem
        self.nlp_prob = {'f': self.obj, 'x': self.opt_variables, 'g': self.g, 'p': self.P}
        self.solver = cas.nlpsol('solver', 'ipopt', self.nlp_prob, self.opts)

        # Equality and inequality constraints
        self.lbg_values = np.zeros(self.g.shape[0])
        self.ubg_values = np.zeros(self.g.shape[0])

        # Set inequality constraints low values to -inf
        self.lbg_values[3*(self.N + 1):] = -float('inf')

        # Inequality constraints value vectors
        self.lbx_values = np.zeros((3*(self.N + 1) + 2*self.N, 1))
        self.ubx_values = np.zeros((3*(self.N + 1) + 2*self.N, 1))

        # Indexes list for states and controls
        self.x_index = np.arange(0, 3*(self.N + 1), 3).tolist()
        self.y_index = np.arange(1, 3*(self.N + 1), 3).tolist()
        self.theta_index = np.arange(2, 3*(self.N + 1), 3).tolist()
        self.v_index = np.arange(3*(self.N + 1), 3*(self.N + 1) + 2*self.N, 2).tolist()
        self.omega_index = np.arange(3*(self.N + 1) + 1, 3*(self.N + 1) + 2*self.N, 2).tolist()

        # Bounds of state constraints (limited to scenario size)
        self.lbx_values[self.x_index, :] = self.scn_size[0]
        self.lbx_values[self.y_index, :] = self.scn_size[1]
        self.lbx_values[self.theta_index, :] = -float('inf')
        self.ubx_values[self.x_index, :] = self.scn_size[2]
        self.ubx_values[self.y_index, :] = self.scn_size[3]
        self.ubx_values[self.theta_index, :] = float('inf')

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

    def calc_distance_fixed_obstacles(self):
        """
        Calculate distance to fixed obstacles.
        """
        self.dist = np.zeros((len(obs.Obstacle.obstacles_x)))
        for k in range(len(obs.Obstacle.obstacles_x)):
            self.dist[k] = cas.sqrt((self.x0[0]-obs.Obstacle.obstacles_x[k])**2 + (self.x0[1]-obs.Obstacle.obstacles_y[k])**2)
        self.min_dist_index = self.dist.argsort()[:self.obstacles]
        self.min_dist_obs = []
        for index in self.min_dist_index:
            self.min_dist_obs.append(obs.Obstacle.obstacles_x[index])
            self.min_dist_obs.append(obs.Obstacle.obstacles_y[index])
            self.min_dist_obs.append(obs.Obstacle.obstacles_diameter[index])

    def calc_distance_other_agvs(self):
        """
        Calculate distance to other AGVs.
        """
        if len(self.agv_list) > 0:
            self.dist_other_agvs = {}
            sorted_dist = {}
            for key in self.other_trajectories.keys():
                msg = self.other_trajectories.get(key)
                x = msg[0].x
                y = msg[0].y
                self.dist_other_agvs[key] = cas.sqrt((self.x0[0]-x)**2 + (self.x0[1]-y)**2)
            sorterd_keys = sorted(self.dist_other_agvs, key=self.dist_other_agvs.get)
            for i in sorterd_keys:
                sorted_dist[i] = self.dist_other_agvs[i]
            self.sorted_dist = sorted_dist

    def start_mpc(self):
        """
        This function updates the values of p and x0 keys from the arguments dictionary (self.args) and then calls the MPC solver. 
        """
        self.mpc_run = True
        self.mpc_start = time.perf_counter()
        if self.x0_update:
            self.x0_update = False
        else:
            self.x0 = cas.reshape(self.X0[1, :], 3, 1)
        self.error = np.linalg.norm(self.x0 - self.xs)
        if self.n > 1:
            self.p = cas.vertcat(self.x0, self.xs, self.col_avoidance_parameters)
            if self.obstacles > 0:
                self.p = cas.vertcat(self.p, self.min_dist_obs)
        else:
            self.p = cas.vertcat(self.x0, self.xs)
            if self.obstacles > 0:
                self.p = cas.vertcat(self.p, self.min_dist_obs)
        self.args.update({'p': self.p})
        self.args.update({'x0': cas.vertcat(cas.reshape(self.X0.T, 3*(self.N + 1), 1), cas.reshape(self.u0.T, 2 * self.N, 1))})
        self.solve_mpc()

    def solve_mpc(self):
        """
        This function calls the solver to obtain the MPC solution with the trajectory and control signals.
        """
        sol = self.solver(x0=self.args['x0'], lbx=self.args['lbx'], ubx=self.args['ubx'], lbg=self.args['lbg'],
                          ubg=self.args['ubg'], p=self.args['p'])
        self.mpc_end = time.perf_counter()
        self.mpc_time = self.mpc_time + (self.mpc_end - self.mpc_start)
        t1 = threading.Thread(target=self.get_x, args=sol['x'].full().T)
        t2 = threading.Thread(target=self.get_u, args=sol['x'].full().T)
        t1.start()
        t2.start()
        self.mpc_iter = self.mpc_iter + 1
        self.mpc_time_avg = self.mpc_time / self.mpc_iter
        if self.save_sim_data:
            self.mpc_time_file.write(f'{self.mpc_end - self.mpc_start}\n')
        t1.join()
        t2.join()

    def get_x(self, sol):
        """
        This function extracts x (predicted states) from the solution to update the self.X0 vector. 
        The predicted states are published as trajectory to local side.
        This function also eliminates the vector's first element and repeats the last one. This variable is used as an 
        initial guess for the next MPC iteration.

        :param sol: MPC solution
        """
        self.sol_x = sol[0:3*(self.N + 1)]
        t = threading.Thread(target=self.publish_trajectory, args=(self.sol_x,))
        t.start()
        self.X0 = cas.reshape(self.sol_x, 3, self.N + 1).T
        self.X0 = cas.vertcat(self.X0[1:, :], self.X0[self.X0.shape[0] - 1, :])

    def get_u(self, sol):
        """
        This function extracts u (control signals) from the solution to update the self.u0 vector. 
        It also eliminates the vector's first element and repeats the last one. This variable is used as an
        initial guess for the next MPC iteration.

        :param sol: MPC solution
        """
        self.sol_u = sol[3*(self.N + 1):]
        self.u0 = cas.reshape(self.sol_u, 2, self.N).T
        self.u0 = cas.vertcat(self.u0[1:, :], self.u0[self.u0.shape[0] - 1, :])

    def publish_trajectory(self, sol_x):
        """
        This functions publishes the AGV trajectory.

        :param sol_x: trajectory solution
        """
        msg = Trajectory()
        trajectory_msg = []
        for k in range(0, 3*(self.N+1), 3):
            pose = Pose()
            pose.x = float(sol_x[k])
            pose.y = float(sol_x[k+1])
            pose.theta = float(sol_x[k+2])
            trajectory_msg.append(pose)
            del pose
        msg.trajectory = trajectory_msg
        msg.agv_id = self.agv_id
        msg.pkg_id = self.pkg_id
        self.trajectory_publisher.publish(msg)
        # if self.mpc_run:
        #     threading.Thread(target=self.append_pkg_id, args=(self.pkg_id,)).start()
        #     self.pkg_id += 1

    def publish_idle_trajectory(self):
        """
        This function publishes the AGV trajectory when its idle. 
        The trajectory in this case is the pose of AGV replicated according to prediction horizon.
        """
        if not self.mpc_run:
            self.publish_trajectory(cas.repmat(self.x0, self.N + 1, 1))
            
    def trajectory_append(self, *args):
        """
        Function to append generated trajectories.
        """
        self.trajectory = np.append(self.trajectory, args)