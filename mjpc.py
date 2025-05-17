import mujoco
import argparse
from mujoco_mpc import agent as agent_lib
import pathlib
import numpy as np
import mediapy as media
import cv2
from time import sleep, perf_counter
import matplotlib.pyplot as plt
from mujoco_mpc import mjpc_parameters
import imageio
from copy import deepcopy

def track_trajectory_quadruped(reference_trajectory: np.ndarray):

    # Get the absolute path to the mujoco_mpc module
    quadruped_path = pathlib.Path(__file__).resolve().parent / "tasks/quadruped/task_flat.xml"

    # quadruped model
    model_path = (
        quadruped_path
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(
        model,
        height=720,
        width=1280)

    links = [
        'FL_calf', 'FL_hip', 'FL_thigh', 
        'FR_calf', 'FR_hip', 'FR_thigh', 
        'HL_hip', 'HR_hip', 'RL_calf', 
        'RL_thigh', 'RR_calf', 'RR_thigh', 
        ]

    model.opt.impratio = 1
    model.opt.timestep = 0.002

    # start_cut = 9000
    # end_cut = 11500
    # interpolated_trajectory = np.linspace(
    #     reference_trajectory[start_cut, :2],
    #     reference_trajectory[end_cut, :2],
    #     end_cut - start_cut
    # )

    # reference_trajectory[start_cut:end_cut, :2] = interpolated_trajectory
    # reference_trajectory = np.vstack([
    #     reference_trajectory[:start_cut, :2],
    #     interpolated_trajectory,
    #     reference_trajectory[end_cut:, :2]
    #     ])
    reference_trajectory = reference_trajectory[:, :2]
    # reference_trajectory += data.mocap_pos[0, :2]
    # reference_trajectory += np.array([0.0, 1.0])
    trajectory_midpoint = np.mean(reference_trajectory, axis=0)

    T = len(reference_trajectory)

    data.qpos[:2] += reference_trajectory[0, :2] - data.mocap_pos[0, :2]
    data.mocap_pos[0, :2] = reference_trajectory[0, :2]

    # trajectories
    qpos = np.zeros((model.nq, T))
    qvel = np.zeros((model.nv, T))
    ctrl = np.zeros((model.nu, T - 1))
    time = np.zeros(T)


    with agent_lib.Agent(
        server_binary_path=pathlib.Path(agent_lib.__file__).parent
        / "mjpc"
        / "ui_agent_server",
        task_id="Quadruped Flat",
        model=model,
    ) as agent:
        agent: agent_lib.Agent
        # costs
        cost_total = np.zeros(T - 1)
        cost_terms = np.zeros((len(agent.get_cost_term_values()), T - 1))

        # # rollout
        # mujoco.mj_resetData(model, data)

        # cache initial state
        qpos[:, 0] = data.qpos
        qvel[:, 0] = data.qvel
        time[0] = data.time

        # frames
        frames = []
        FPS = 1.0 / 0.02

        camera = mujoco.MjvCamera()
        camera.lookat = np.array([trajectory_midpoint[0], trajectory_midpoint[1], 0.0])
        camera.distance = 8.0
        camera.elevation = -90.0

        skip = 10
        render_skip = 10

        # agent.set_cost_weights({
        #     "Balance": 1.0,
        # })
        print(agent.get_all_modes())

        print(agent.get_cost_weights())

        print(agent.get_state())

        data_copy = None

        # simulate
        for t in range(0, T - 1):
            start = perf_counter()

            # if t == 7500:
            #     data_copy = deepcopy(data)
            #     mujoco.mj_resetData(model, data_copy)

            # print(data.mocap_pos)   
            # set planner state

            data.mocap_pos[0, :2] = reference_trajectory[t, :2]

            agent.set_mocap({"goal": mjpc_parameters.Pose(data.mocap_pos, data.mocap_quat[0])})

            agent.set_state(mocap_pos=data.mocap_pos)

            # if t % skip == 0:
            #     print("t = ", t * model.opt.timestep)
            #     # agent.set_state(mocap_pos=data.mocap_pos)
            #     agent.set_state(
            #         time=data.time,
            #         qpos=data.qpos,
            #         qvel=data.qvel,
            #         act=data.act,
            #         mocap_pos=data.mocap_pos,
            #         mocap_quat=data.mocap_quat,
            #         userdata=data.userdata,
            #     )

            #     agent.planner_step()

            # # set ctrl from agent policy
            # data.ctrl = agent.get_action()

            # ctrl[:, t] = data.ctrl

            # # get costs
            cost_total[t] = agent.get_total_cost()
            # for i, c in enumerate(agent.get_cost_term_values().items()):
            #     cost_terms[i, t] = c[1]

            # # step
            # mujoco.mj_step(model, data)

            state = agent.get_state()

            # # # cache
            qpos[:, t + 1] = state.qpos
            qvel[:, t + 1] = state.qvel
            time[t + 1] = state.time

            # if t % render_skip == 0:
            #     try:
            #         # render and save frames
            #         renderer.update_scene(data, camera)
            #         pixels = renderer.render()
            #         frames.append(pixels)

            #         # Display the video using OpenCV
            #         cv2.imshow("Simulation", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
            #         if cv2.waitKey(1) == ord('q'):
            #             break
            #     except Exception as e:
            #         pass
            
            end = perf_counter() - start
            if end < model.opt.timestep:
                sleep(model.opt.timestep - end)
            # print(perf_counter() - start)

    # Save the video using imageio

    # output_video_path = "simulation_output.mp4"
    # imageio.mimwrite(output_video_path, frames, fps=FPS, quality=8)
    # print(f"Video saved to {output_video_path}")

    # Plot the total cost over real time
    real_time = np.arange(len(cost_total)) * model.opt.timestep


    # cv2.destroyAllWindows()

    return real_time, cost_total, qpos

def plot_tracked_trajectory(real_time, cost_total, qpos, reference_trajectory, rectangles):

    offset = np.array([0.3, 0.0])
    plt.figure(figsize=(10, 6))
    plt.plot(real_time, cost_total, label="Total Cost")
    plt.xlabel("Real Time (seconds)")
    plt.ylabel("Cost")
    plt.title("Total Cost Over Real Time")
    plt.legend()
    plt.grid()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot qpos[:, :2] and reference_trajectory
    ax.plot(qpos[0, :], qpos[1, :], label="Quadruped Position", color="blue")
    ax.plot(reference_trajectory[:, 0], reference_trajectory[:, 1], label="Reference Trajectory", color="red", linestyle="--")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Quadruped Position vs Reference Trajectory")
    ax.legend()
    ax.grid()

    # Plot all rectangular obstacles
    colors = ['blue', 'black', 'gray', 'red', 'green', 'purple', 'orange', 'brown']
    for i, rect in enumerate(rectangles):
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
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj-path', default='trajectory.txt', type=str, help='Path to the trajectory file')
    args = parser.parse_args()
    reference_trajectory = np.loadtxt(args.traj_path, delimiter=',')

    example_objects = [
        [0, 'sofa', 0.5, 0.5, 1.5, 2.5],
        [1, 'tv', 3.8, 1.25, 4.2, 1.75],
    ]

    real_time, cost_total, qpos = track_trajectory_quadruped(reference_trajectory)
    traj_path: str = args.traj_path
    cost_path = f'{traj_path.split(".")[0]}_cost.txt'
    qpos_path = f'{traj_path.split(".")[0]}_qpos.txt'
    np.savetxt(cost_path, cost_total)
    np.savetxt(qpos_path, qpos)

    plot_tracked_trajectory(real_time, cost_total, qpos, reference_trajectory, example_objects)