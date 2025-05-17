import numpy as np
import argparse

from llm_query import plan_trajectory
from collision_avoidance_optimization import optimize_trajectory
from trajectory_tools import interpolate_trajectory, get_trajectory_length
from mjpc import track_trajectory_quadruped, plot_tracked_trajectory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--enforce-full-sparse-trajectory-collision',
        action='store_true',
        help='Enforce full sparse trajectory collision avoidance'
    )
    parser.add_argument(
        '--traj-save-name',
        default='trajectory',
        type=str,
        help='Trajectory save name'
    )
    parser.add_argument(
        '--robot-start',
        default=[0.0, 0.0, 0.0],
        type=float,
        nargs=3,
        help='Starting position of the robot in the format x y theta'
    )
    args = parser.parse_args()

    optimization_timestep = 0.1
    final_timestep = 0.002
    maximum_velocity = 0.25

    enforce_full_sparse_trajectory_collision = True

    example_objects = [
        [0, 'sofa', 0.5, 0.5, 1.5, 2.5],
        [1, 'tv', 3.8, 1.25, 4.2, 1.75],
    ]
    robot_start = np.array(args.robot_start)[None, :]
    user_command = input("Enter a command: ")
    save_path = args.traj_save_name
    
    trajectory = plan_trajectory(user_command, example_objects, robot_start)
    print("Planned trajectory:", trajectory)


    optimized_trajectory, interpolated_trajectory = optimize_trajectory(
        robot_start[0], 
        np.array(trajectory)[:-1], 
        example_objects, 
        optimization_timestep,
        final_timestep,
        maximum_velocity,
        enforce_full_sparse_trajectory_collision=enforce_full_sparse_trajectory_collision)

    with open(f'{save_path}_optimized.txt', "w") as f:
        np.savetxt(f, optimized_trajectory, fmt='%s', delimiter=',')
    print("Optimized trajectory saved to", save_path)
    with open(f'{save_path}_interpolated.txt', "w") as f:
        np.savetxt(f, interpolated_trajectory, fmt='%s', delimiter=',')
    print("Interpolated trajectory saved to", save_path)
    with open(f'{save_path}_planned.txt', "w") as f:
        np.savetxt(f, trajectory, fmt='%s', delimiter=',')
    print("LLM Planned trajectory saved to", save_path)

    real_time, cost_total, qpos = track_trajectory_quadruped(reference_trajectory=interpolated_trajectory)
    plot_tracked_trajectory(real_time, cost_total, qpos, optimized_trajectory, example_objects)

if __name__ == "__main__":
    main()