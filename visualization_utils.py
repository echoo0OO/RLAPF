import matplotlib.pyplot as plt


# --- 新增：绘图辅助函数 ---

def plot_trajectory(trajectory, sensors_true_pos, save_path, area_size):
    """绘制无人机飞行轨迹和传感器真实位置"""
    traj_arr = np.array(trajectory)
    sensors_arr = np.array(sensors_true_pos)

    plt.figure(figsize=(8, 8))
    plt.plot(traj_arr[:, 0], traj_arr[:, 1], 'b-', label='UAV Trajectory', alpha=0.7)
    plt.scatter(traj_arr[0, 0], traj_arr[0, 1], c='green', marker='o', s=100, label='Start', zorder=5)
    plt.scatter(traj_arr[-1, 0], traj_arr[-1, 1], c='red', marker='x', s=100, label='End', zorder=5)
    plt.scatter(sensors_arr[:, 0], sensors_arr[:, 1], c='purple', marker='^', s=120, label='Sensors (True Pos)',
                zorder=5)

    plt.title('Drone Trajectory and Sensor Locations')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.xlim(0, area_size[0])
    plt.ylim(0, area_size[1])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(save_path)
    plt.close()  # 关闭图形，防止在循环中消耗过多内存


def plot_remaining_data(comm_log, num_sensors, total_data, save_path):
    """绘制每个传感器剩余数据量的变化"""
    if not comm_log:
        return  # 如果没有通信日志，则不绘图

    plt.figure(figsize=(10, 6))

    # 为每个传感器的数据创建一个时间线
    steps = [log['step'] for log in comm_log]
    data_over_time = np.array([log['remaining_data'] for log in comm_log])

    for i in range(num_sensors):
        plt.plot(steps, data_over_time[:, i] / 1e6, label=f'Sensor {i}')

    plt.title('Remaining Data per Sensor Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Remaining Data (Mbits)')
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_position_error(loc_log, sensors_true_pos, save_path):
    """绘制估计位置与真实位置的误差变化"""
    if not loc_log:
        return  # 如果没有定位更新日志，则不绘图

    plt.figure(figsize=(10, 6))
    update_counts = range(len(loc_log))  # 横坐标是更新次数

    errors_over_updates = []
    for log_entry in loc_log:
        est_positions = log_entry['est_positions']
        errors = np.linalg.norm(est_positions - sensors_true_pos, axis=1)
        errors_over_updates.append(errors)

    errors_arr = np.array(errors_over_updates)

    for i in range(errors_arr.shape[1]):
        plt.plot(update_counts, errors_arr[:, i], label=f'Sensor {i} Error')

    plt.title('Position Estimation Error vs. Localization Updates')
    plt.xlabel('Number of Localization Updates')
    plt.ylabel('Euclidean Distance Error (m)')
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_uncertainty_radius(loc_log, save_path):
    """绘制估计半径的变化"""
    if not loc_log:
        return  # 如果没有定位更新日志，则不绘图

    plt.figure(figsize=(10, 6))
    update_counts = range(len(loc_log))  # 横坐标是更新次数

    radii_over_updates = np.array([log['est_radii'] for log in loc_log])

    for i in range(radii_over_updates.shape[1]):
        plt.plot(update_counts, radii_over_updates[:, i], label=f'Sensor {i} Radius')

    plt.title('Uncertainty Radius vs. Localization Updates')
    plt.xlabel('Number of Localization Updates')
    plt.ylabel('Uncertainty Radius (m)')
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(save_path)
    plt.close()