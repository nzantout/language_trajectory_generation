from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Dict, Any
import casadi as ca
from scipy.interpolate import CubicSpline
from collections import deque
from collision_tools import enforce_collision_avoidance
from trajectory_tools import interpolate_trajectory, get_trajectory_length

class TrajectoryOptimizer:
    def __init__(self, 
                 start_state: np.ndarray,
                 goal_state: np.ndarray,
                 rectangles: List[List[Any]] = None,
                 dt: float = 0.1,
                 maximum_velocity: float = 1.0,
                 collision_buffer_distance: float = 0.3):
        """
        Initialize the trajectory optimizer.
        
        Args:
            start_state: Initial state vector [x, y, theta, v, omega, a]
            goal_state: Target state vector [x, y, theta, v, omega, a]
            rectangles: List of rectangles, each defined as [id, name, x_min, y_min, x_max, y_max]
            dt: Time step for discretization
        """
        self.start_state = start_state
        self.goal_state = goal_state
        self.state_dim = 6
        self.control_dim = 2  # [acceleration, angular_velocity]
        
        # Default rectangles if none provided
        self.rectangles = rectangles if rectangles is not None else [
            [1, 'Sofa', 0.6, 0.8, 1.4, 1.4],  # First rectangle
            [2, 'TV', 1.6, 0.4, 2.4, 1.0],  # Second rectangle
            [3, 'Table', 0.2, 1.6, 1.0, 2.2]   # Third rectangle
        ]

        # Define parameters
        self.dt = dt
        self.buffer_distance = collision_buffer_distance  # Minimum distance to maintain from obstacles
        self.maximum_velocity = maximum_velocity  # Maximum velocity constraint
        self.maximum_omega = np.inf # Maximum angular velocity constraint
        self.maximum_acceleration = maximum_velocity
        self.minimum_acceleration = -maximum_velocity
        self.maximum_alpha = np.inf
        self.maximum_jerk = maximum_velocity
        self.minimum_jerk = -maximum_velocity
        
        # Initialize CasADi variables
        self.opti = ca.Opti()
        
    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        System dynamics function.
        
        Args:
            state: Current state [x, y, theta, v, omega]
            control: Control input [acceleration, angular_velocity]
            
        Returns:
            Next state after applying control
        """
        # Use indexing instead of unpacking for CasADi compatibility
        x = state[0]
        y = state[1]
        theta = state[2]
        v = state[3]
        omega = state[4]
        a = state[5]
        
        jerk = control[0]
        alpha = control[1]
        
        # Simple unicycle model
        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = omega
        dv = a
        da = jerk
        domega = alpha
        
        return ca.vertcat(dx, dy, dtheta, dv, da, domega)

    def hermite_simpson(self, x1: np.ndarray, x2: np.ndarray, u: np.ndarray):
        fx1 = self.dynamics(x1, u)
        fx2 = self.dynamics(x2, u)
        x12 = (x1 + x2) / 2 + self.dt / 8 * (fx1 - fx2)
        fx12 = self.dynamics(x12, u)

        return x1 + self.dt / 6 * (fx1 + 4 * fx12 + fx2) - x2
    
    def objective_function(self, X: ca.MX, U: ca.MX, reference_trajectory: np.ndarray) -> float:
        """
        Objective function to minimize.
        
        Args:
            X: State trajectory matrix
            U: Control trajectory matrix
            reference_trajectory: Reference trajectory array
        Returns:
            Cost value to minimize
        """ 
        N = X.shape[1]

        Q = ca.DM.eye(self.state_dim)
        # Q[5, 5] = 10
        R = 0.1*ca.DM.eye(self.control_dim) 
        Qf = ca.DM.eye(self.state_dim)

        X_ref = reference_trajectory[:, :self.state_dim]
        U_ref = reference_trajectory[:, self.state_dim:]

        J = 0
        for k in range(N-1):
            state_error = X[:, k] - X_ref[k]
            control_error = U[:, k] - U_ref[k]
            J += 0.5 * ca.mtimes([state_error.T, Q, state_error])
            J += 0.5 * ca.mtimes([control_error.T, R, control_error])

        final_error = X[:, -1] - self.goal_state[:self.state_dim]
        J += 0.5 * ca.mtimes([final_error.T, Qf, final_error])

        return J
    
    def dynamics_constraint(self, X: ca.MX, U: ca.MX) -> ca.MX:
        """
        Dynamics constraint.
        """
        # Dynamics constraints using Hermite-Simpson
        N = X.shape[1]

        c = ca.MX.zeros(self.state_dim * (N-1), 1)

        for k in range(N-1):
            x1 = X[:, k]
            x2 = X[:, k+1]
            u = U[:, k]
            
            # Hermite-Simpson constraints
            c[self.state_dim * k:self.state_dim * (k+1)] = self.hermite_simpson(x1, x2, u)
        
        return c
        
    def collision_constraint(self, X: ca.MX, U: ca.MX) -> ca.MX:
        """
        Collision avoidance constraint for multiple rectangular obstacles.
        
        Args:
            X: State trajectory matrix
            U: Control trajectory matrix
            
        Returns:
            Minimum distance to obstacles (should be positive)
        """
        N = X.shape[1]
        c = ca.MX.zeros(N, 1)

        buffer_distance = 0.3

        for k in range(N):
            # Get robot position at timestep k
            pos = X[:2, k]
            
            # Initialize minimum distance to a large value
            min_dist = 1e6
            
            # Check distance to each rectangle
            for rect in self.rectangles:
                # Extract coordinates from rectangle (skip the id and name)
                x_min, y_min, x_max, y_max = rect[2:]
                
                # Calculate signed distances to each edge of rectangle
                dx = ca.fmax(x_min - buffer_distance - pos[0], pos[0] - (x_max + buffer_distance))
                dy = ca.fmax(y_min - buffer_distance - pos[1], pos[1] - (y_max + buffer_distance))
                
                # Distance to this rectangle
                dist = ca.fmax(dx, dy)
                
                # Update minimum distance
                min_dist = ca.fmin(min_dist, dist)
            
            c[k] = min_dist
        
        return c
    
    def equality_constraints(self, X: ca.MX, U: ca.MX) -> ca.MX:
        """
        Equality constraints.
        """
        ci = X[:, 0] - self.start_state[:self.state_dim]
        cf = X[:, -1] - self.goal_state[:self.state_dim]
        dc = self.dynamics_constraint(X, U)

        c = ca.vertcat(ci, cf, dc)

        return c

        
    def inequality_constraints(self, X: ca.MX, U: ca.MX) -> ca.MX:
        """
        Inequality constraints.
        """
        c = self.collision_constraint(X, U)

        return c
 
    
    def optimize(self, reference_trajectory: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform trajectory optimization using CasADi and IPOPT.
        
        Returns:
            Optimized trajectory and optimization info
        """
        # Get dimensions
        N = len(reference_trajectory)  # number of time steps
        
        # Define variables
        X = self.opti.variable(self.state_dim, N)  # states
        U = self.opti.variable(self.control_dim, N)  # controls
    
        
        # Set objective
        cost = self.objective_function(X, U, reference_trajectory)
        self.opti.minimize(cost)
        
        # Equality constraints
        eq_cons = self.equality_constraints(X, U)
        self.opti.subject_to(eq_cons == 0)
        
        # Inequality constraints
        ineq_cons = self.inequality_constraints(X, U)
        self.opti.subject_to(ineq_cons >= 0)
        

        # Control bounds
        for k in range(N):
            # State bounds [x, y, theta, v, omega]
            self.opti.subject_to(self.opti.bounded(-np.inf, X[0, k], np.inf))  # x
            self.opti.subject_to(self.opti.bounded(-np.inf, X[1, k], np.inf))  # y
            # self.opti.subject_to(self.opti.bounded(-np.pi, X[2, k], np.pi))    # theta
            self.opti.subject_to(self.opti.bounded(0, X[3, k], self.maximum_velocity))          # v
            self.opti.subject_to(self.opti.bounded(-self.maximum_omega, X[4, k], self.maximum_omega))       # omega
            self.opti.subject_to(self.opti.bounded(self.minimum_acceleration, X[5, k], self.maximum_acceleration))       # a
            
            
            # # Control bounds [acceleration, angular_velocity]
            self.opti.subject_to(self.opti.bounded(self.minimum_jerk, U[0, k], self.maximum_jerk))       # jerk
            self.opti.subject_to(self.opti.bounded(-self.maximum_alpha, U[1, k], self.maximum_alpha))       # angular_velocity
        
        # Set initial guess
        initial_guess = reference_trajectory
        self.opti.set_initial(X, initial_guess[:, :self.state_dim].T)
        self.opti.set_initial(U, initial_guess[:, self.state_dim:].T)
        
        # Set solver options
        opts = {
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 4,
            'print_time': 1,
            'ipopt.acceptable_tol': 1e-6,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }
        
        # Solve the optimization problem
        try:
            self.opti.solver('ipopt', opts)
            sol = self.opti.solve()
            
            # Extract solution
            X_sol = sol.value(X)
            U_sol = sol.value(U)
            
            # Combine states and controls
            trajectory = np.vstack((X_sol, U_sol)).T
            
            # Create result dictionary similar to scipy.optimize
            result = {
                'success': True,
                'fun': sol.value(cost),
                'message': 'Optimization successful',
                'status': 0
            }
            
            return trajectory, result
            
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return None, {'success': False, 'message': str(e)}
    
    def plot_trajectory(
            self, 
            ax, 
            states: np.ndarray = None,
            reinterpolated_trajectory: np.ndarray = None,
            reference_trajectory: np.ndarray = None, 
            sparser_trajectory: np.ndarray = None,
            tracked_trajectory: np.ndarray = None):
        """Plot the xy trajectory with obstacles."""
        if reference_trajectory is not None:
            ref_traj = reference_trajectory.reshape(-1, self.state_dim + self.control_dim)
            ref_states = ref_traj[:, :self.state_dim]
            ax.plot(ref_states[:, 0], ref_states[:, 1], 'g--', linewidth=2, label='Interpolated Reference Trajectory')
        
        if sparser_trajectory is not None:
            ax.scatter(sparser_trajectory[:, 0], sparser_trajectory[:, 1], c='g', s=100, label='LLM Generated Trajectory')
        
        if reinterpolated_trajectory is not None:
            reinterpolated_traj = reinterpolated_trajectory.reshape(-1, self.state_dim + self.control_dim)
            reinterpolated_states = reinterpolated_traj[:, :self.state_dim]
            ax.plot(reinterpolated_states[:, 0], reinterpolated_states[:, 1], 'r--', linewidth=2, label='Reinterpolated Trajectory')
        
        if tracked_trajectory is not None:
            ax.plot(tracked_trajectory[0, :], tracked_trajectory[1, :], 'r-', linewidth=2, label='Quadruped Tracked Trajectory')
            
        if states is not None:
            ax.plot(states[:, 0], states[:, 1], 'b-', label='Optimized Trajectory')
            ax.scatter(sparser_trajectory[0, 0], sparser_trajectory[0, 1], c='b', s=200, label='Start')
            ax.scatter(sparser_trajectory[-1, 0], sparser_trajectory[-1, 1], c='r', s=200, label='Goal')
        
        # Plot all rectangular obstacles
        colors = ['blue', 'black', 'gray', 'red', 'green', 'purple', 'orange', 'brown']
        for i, rect in enumerate(self.rectangles):
            rect_id, name, x_min, y_min, x_max, y_max = rect
            width = x_max - x_min
            height = y_max - y_min
            
            color = colors[i % len(colors)]
                
            rect_patch = plt.Rectangle((x_min, y_min),
                                     width, height, 
                                     color=color, alpha=0.3,
                                     label=f"{name} (ID: {rect_id})")
            ax.add_patch(rect_patch)
        
        ax.grid(True)
        ax.axis('equal')
        ax.legend()
        ax.set_title('XY Trajectory')

    def plot_trajectory_parameter(self, ax, parameter_name: str, trajectory: np.ndarray, reference_trajectory: np.ndarray = None):
        """Plot a state or control."""
        parameter_dict = {
            'X Position': 0,
            'Y Position': 1,
            'Theta': 2,
            'Velocity': 3,
            'Angular Velocity': 4,
            'Acceleration': 5,
            'Jerk': 6,
            'Angular Acceleration': 7
        }
        parameter_units = [
            'm',
            'm',
            'rad',
            'm/s',
            'rad/s',
            'm/s^2',
            'm/s^3',
            'rad/s^2'
        ]
        parameter_idx = parameter_dict[parameter_name]
        time = np.arange(len(trajectory)) * self.dt
        ax.scatter(time, trajectory[:, parameter_idx], c='b', s=1, label=f'Optimized {parameter_name}')
        if reference_trajectory is not None and reference_trajectory.shape[1] > parameter_idx:
            ref_traj = reference_trajectory.reshape(-1, self.state_dim + self.control_dim)
            ax.plot(time, ref_traj[:, parameter_idx], 'g--', label=f'Reference {parameter_name}')
        ax.grid(True)
        ax.legend()
        ax.set_title(f'{parameter_name} vs Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{parameter_name} ({parameter_units[parameter_idx]})')
    
    def visualize_trajectory(
            self, 
            trajectory: np.ndarray,
            reinterpolated_trajectory: np.ndarray = None,
            reference_trajectory: np.ndarray = None, 
            sparser_trajectory: np.ndarray = None):
        """
        Visualize the optimized trajectory.
        
        Args:
            trajectory: Optimized trajectory array
            reference_trajectory: Optional reference trajectory array
        """
        traj = trajectory.reshape(-1, self.state_dim + self.control_dim)
        
        # Create three separate figures
        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111)
        self.plot_trajectory(ax1, traj, reinterpolated_trajectory, reference_trajectory, sparser_trajectory)
        plt.show()

        # fig2 = plt.figure(figsize=(10, 6))
        # ax2 = fig2.add_subplot(111)
        # self.plot_trajectory_parameter(ax2, 'Acceleration', traj, reference_trajectory)
        # plt.show()

        # fig2 = plt.figure(figsize=(10, 6))
        # ax2 = fig2.add_subplot(111)
        # self.plot_trajectory_parameter(ax2, 'Angular Velocity', traj, reference_trajectory)
        # plt.show()

        # fig3 = plt.figure(figsize=(10, 6))
        # ax3 = fig3.add_subplot(111)
        # self.plot_theta(ax3, states, reference_trajectory)
        # plt.show()

def optimize_trajectory(
        initial_coords: np.ndarray, 
        sparse_trajectory_xy: np.ndarray, 
        rectangles: list, 
        optimization_timestep: float, 
        final_timestep: float,
        maximum_velocity: float,
        enforce_full_sparse_trajectory_collision: bool = False) -> np.ndarray:

    collision_buffer_distance = 0.3

    if enforce_full_sparse_trajectory_collision:
        for i in range(len(sparse_trajectory_xy)):
            sparse_trajectory_xy[i] = enforce_collision_avoidance(sparse_trajectory_xy[i], rectangles, collision_buffer_distance + 1e-3)
    else:
        sparse_trajectory_xy[-1] = enforce_collision_avoidance(sparse_trajectory_xy[-1], rectangles, collision_buffer_distance + 1e-3)

    # Calculate the total time based on the trajectory length and maximum velocity
    T = get_trajectory_length(sparse_trajectory_xy) / (maximum_velocity)

    print(f"Total time: {T:.2f} seconds")

    # Define start and goal states
    start_state = np.array([initial_coords[0], initial_coords[1], initial_coords[2], 0.0, 0.0, 0.0, 0.0, 0.0])  # [x, y, theta, v, omega, a]
    goal_state = np.array([sparse_trajectory_xy[-1, 0], sparse_trajectory_xy[-1, 1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    num_steps = int(T/optimization_timestep)

    # Using numpy's linspace for linear interpolation
    sparse_trajectory_other = np.array([
        np.linspace(start_state[i], goal_state[i], len(sparse_trajectory_xy)) 
        for i in range(2, len(start_state))
    ]).T

    sparse_trajectory = np.hstack((sparse_trajectory_xy, sparse_trajectory_other))

    reference_trajectory = interpolate_trajectory(sparse_trajectory, num_steps)

    optimizer = TrajectoryOptimizer(
        start_state, 
        goal_state,
        rectangles=rectangles,
        dt=optimization_timestep,
        maximum_velocity=maximum_velocity,
        collision_buffer_distance=collision_buffer_distance)
    
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111)
    optimizer.plot_trajectory(ax1, reference_trajectory=reference_trajectory, sparser_trajectory=sparse_trajectory)
    plt.show()
    
    trajectory, result = optimizer.optimize(reference_trajectory)

    # Print optimization results
    print(f"Optimization successful: {result['success']}")
    print(f"Final cost: {result['fun']}")

    final_optimized_trajectory = interpolate_trajectory(trajectory, int(T/final_timestep))

    reinterpolated_reference_trajectory = interpolate_trajectory(reference_trajectory, int(T/final_timestep))
    
    # Visualize trajectory
    optimizer.visualize_trajectory(
        final_optimized_trajectory, 
        # final_optimized_trajectory, 
        reference_trajectory=reference_trajectory, 
        sparser_trajectory=sparse_trajectory)

    return final_optimized_trajectory, reinterpolated_reference_trajectory


# Example usage
if __name__ == "__main__":
    # Create reference trajectory by linear interpolation
    dt = 0.1  # timestep
    T = 10.0   # total time
    

    sparser_trajectory_xy = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [1.5, 0.0],
        [1.5, 0.5],
        [1.5, 1.0],
        [1.5, 1.5],
        [1.5, 2.0],
        [1.5, 2.5],
        [2.0, 2.25],
        [2.5, 2.0],
        [3.0, 1.75],
        [3.5, 1.5],
    ])

    rectangles = [
        [1, 'Sofa', 0.5, 0.5, 1.5, 2.5],
        [2, 'TV', 3.8, 1.25, 4.2, 1.75],
    ]

    robot_coords = np.array([0.0, 0.0, np.pi/2])

    optimize_trajectory(robot_coords, sparser_trajectory_xy, rectangles, dt, T)

