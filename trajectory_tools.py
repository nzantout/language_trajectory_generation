import numpy as np
from scipy.interpolate import CubicSpline

def interpolate_trajectory(trajectory: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Interpolate a trajectory using scipy's cubic spline interpolation.
    
    Args:
        trajectory: Original trajectory array of shape (N, state_dim + control_dim)
        num_samples: Number of samples in the interpolated trajectory
        
    Returns:
        Interpolated trajectory array of shape (num_samples, state_dim + control_dim)
    """
    # Get original time points
    t_original = np.arange(len(trajectory))
    
    # Create new time points for interpolation
    t_new = np.linspace(0, len(trajectory)-1, num_samples)
    
    # Interpolate each dimension separately
    trajectory_interp = np.zeros((num_samples, trajectory.shape[1]))
    for i in range(trajectory.shape[1]):
        cs = CubicSpline(t_original, trajectory[:, i])
        trajectory_interp[:, i] = cs(t_new)
    
    return trajectory_interp

def get_trajectory_length(trajectory: np.ndarray) -> float:
    """
    Integrate a trajectory using the trapezoidal rule.
    
    Args:
        trajectory: XY trajectory array of shape (N, 2)
        
    Returns:
        Trajectory length as a float
    """
    trajectory_lengths = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)
    return np.sum(trajectory_lengths)