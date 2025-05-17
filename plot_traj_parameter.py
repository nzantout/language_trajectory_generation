import matplotlib.pyplot as plt
import numpy as np
from collision_avoidance_optimization import TrajectoryOptimizer
from collision_tools import enforce_collision_avoidance

rectangles = [
    [0, 'sofa', 0.5, 0.5, 1.5, 2.5],
    [1, 'tv', 3.8, 1.25, 4.2, 1.75],
]
trajopt = TrajectoryOptimizer(np.zeros(8), np.zeros(8), rectangles, dt=0.002)

string = "forward_around_close"
# optimized_trajectory = np.loadtxt(f'{string}_optimized.txt', delimiter=',')
# interpolated_trajectory = np.loadtxt(f'{string}_interpolated.txt', delimiter=',')
# planned_trajectory = np.loadtxt(f'{string}_planned.txt', delimiter=',')
# tracked_trajectory = np.loadtxt(f'{string}_optimized_qpos.txt', delimiter=' ')

# planned_trajectory = np.array([enforce_collision_avoidance(planned_trajectory[i], rectangles) for i in range(len(planned_trajectory))])


# fig, ax = plt.subplots()
# trajopt.plot_trajectory(
#     ax,
#     optimized_trajectory,
#     reference_trajectory=interpolated_trajectory,
#     sparser_trajectory=planned_trajectory,
#     tracked_trajectory=tracked_trajectory,
# )

# # Show the plot
# plt.show()

cost_total = np.loadtxt(f'{string}_optimized_cost.txt', delimiter=',')
real_time = np.arange(len(cost_total)) * 0.002
plt.figure(figsize=(15, 9))
plt.plot(real_time, cost_total, label="Total Cost")
plt.xlabel("Real Time (seconds)")
plt.ylabel("Cost")
plt.title("Total Cost Over Real Time")
plt.legend()
plt.grid()
plt.savefig(f'{string}_cost.png')
plt.show()