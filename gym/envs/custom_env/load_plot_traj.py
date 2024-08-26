import numpy as np
import matplotlib.pyplot as plt

class TrajectoryPlotter:
    def __init__(self, filename, m, n, d, epsilon, r_c, seed):
        self.filename = filename
        self.m = m  # Number of UAVs
        self.n = n  # Number of Targets
        self.d = d  # Coverage distance
        self.epsilon = epsilon  # Coverage gap
        self.r_c = r_c  # Charge station radius
        self.seed = seed
        self.trajectory_data = None

    def load_trajectories(self):
        self.trajectory_data = np.load(self.filename)
        print(f"Loaded trajectories from {self.filename}")

    def plot_trajectory(self, policy):
        plt.figure(figsize=(10, 6))
        plt.axis('equal')

        # Draw targets and their coverage bands
        for target_idx in range(self.n):
            target_data = self.trajectory_data[target_idx]
            target_color = plt.cm.spring(target_idx / self.n)
            last_target_x, last_target_y = target_data[-1]
            plt.plot(last_target_x, last_target_y, marker='$\u2691$', color=target_color, markersize=15,
                     label=f'Target {target_idx+1}', linestyle='None', zorder=5)
            coverage_band_outer = plt.Circle((last_target_x, last_target_y), self.d + self.epsilon,
                                             color=target_color, alpha=0.3)
            coverage_band_inner = plt.Circle((last_target_x, last_target_y), self.d - self.epsilon,
                                             color='white', alpha=1)
            plt.gca().add_patch(coverage_band_outer)
            plt.gca().add_patch(coverage_band_inner)
            pre_target_x, pre_target_y = target_data[0]
            for target_x, target_y in target_data:
                plt.plot([pre_target_x, target_x], [pre_target_y, target_y], color=target_color)
                pre_target_x, pre_target_y = target_x, target_y

        # Draw the charging station
        charging_station = plt.Circle((0, 0), self.r_c, color='blue', fill=True, alpha=0.3, label='Charging Station')
        plt.gca().add_patch(charging_station)
        plt.plot(0, 0, marker='$\u26A1$', color='yellow', markersize=20, linestyle='None', zorder=5)

        # Draw UAV trajectories
        for uav_idx in range(self.m):
            uav_data = self.trajectory_data[self.n + uav_idx]
            prev_x, prev_y = uav_data[0]
            for x, y in uav_data:
                plt.plot([prev_x, x], [prev_y, y], color='black')
                prev_x, prev_y = x, y
            last_x, last_y = uav_data[-1]
            plt.plot(last_x, last_y, marker='$\u2708$', color='red', markersize=10, linestyle='None', zorder=5)

        limit = 60
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)
        plt.legend(loc='upper right')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'UAV Trajectory for Policy {policy}')
        plt.savefig(f'PLOT/real_UAV{self.m}_Target{self.n}_{policy}_trajectory_{self.seed}.png')
        plt.show()

if __name__ == "__main__":
    filename = 'traj/real_target_trajectory_m:4_n:2_seed:0.npy'  # Example filename
    m = 4  # Number of UAVs
    n = 2  # Number of Targets
    d = 40  # Coverage distance
    epsilon = 2  # Coverage gap
    r_c = 10  # Charge station radius
    seed = 0  # Seed for reproducibility

    plotter = TrajectoryPlotter(filename, m, n, d, epsilon, r_c, seed)
    plotter.load_trajectories()
    plotter.plot_trajectory(policy='Loaded')
