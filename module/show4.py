import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm

# --- 字体设置 (与之前相同) ---
try:
    font_path_candidates = [
        '/System/Library/Fonts/PingFang.ttc',
        'C:/Windows/Fonts/msyh.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        'SimHei.ttf',
        'Arial Unicode MS'
    ]
    font_path = None
    for candidate in font_path_candidates:
        try:
            fm.FontProperties(fname=candidate if '/' in candidate or '\\' in candidate else candidate)
            font_path = candidate
            break
        except Exception:
            continue
    if font_path:
        print(f"Using font: {font_path}")
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    else:
        print("Chinese font not found, using default. Labels might not display correctly.")
        plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Error setting Chinese font: {e}. Using default font.")
    plt.rcParams['axes.unicode_minus'] = False


def calculate_dynamic_attraction_reward(p_uav_x, p_uav_y, p_est_x, p_est_y, u_rad, A, k, C1, C2, delta):
    distance = np.sqrt((p_uav_x - p_est_x)**2 + (p_uav_y - p_est_y)**2)
    R_base_attraction = A * np.exp(-k * distance)
    uncertainty_modulator = (C1 / (C2 + u_rad)) + delta # 确保 u_rad 不会使分母为0
    R_total = R_base_attraction * uncertainty_modulator
    return R_total

# --- 1. 奖励函数和可视化参数 ---
A = 100.0
k = 0.2
C1 = 1.0
# C2 = 1.0   # 旧的 C2 值
C2 = 0.2    # 修改 C2 以增加敏感度
delta = 0.05

p_est_x_origin = 0.0
p_est_y_origin = 0.0

x_range = np.linspace(-15, 15, 100)
y_range = np.linspace(-15, 15, 100)
X, Y = np.meshgrid(x_range, y_range)

u_rad_values = [0.5, 2.0, 8.0] # 可以尝试更宽范围的U_rad, e.g., [0.1, 1.0, 5.0, 10.0]

# --- 2. 计算全局Z轴范围 ---
z_max_overall = 0
# 找到在所有u_rad条件下，奖励的理论最大值 (发生在distance=0)
# (C2 + u_rad) 不能为0, u_rad通常为正
if min(u_rad_values) + C2 <= 0: # 简单检查避免分母为0或负
    print("警告: C2 + min(u_rad_values) 可能导致分母为零或负，请检查参数！")
    # 使用一个基于A和delta的估算最大值，或者抛出错误
    z_max_overall = A * (C1 / (C2 + min(u_rad_values) + 1e-6) + delta) # 加一个极小值避免除零
else:
    z_max_overall = A * ( (C1 / (C2 + min(u_rad_values))) + delta )


# --- 3. 创建和显示三维图像 ---
fig = plt.figure(figsize=(18, 7)) # 调整图像总尺寸，可能需要给z轴标签更多空间
plot_num = len(u_rad_values)

for i, u_rad_val in enumerate(u_rad_values):
    Z = calculate_dynamic_attraction_reward(X, Y, p_est_x_origin, p_est_y_origin,
                                            u_rad_val, A, k, C1, C2, delta)
    ax = fig.add_subplot(1, plot_num, i + 1, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', vmin=0, vmax=z_max_overall*1.05) # 添加 vmin, vmax

    ax.set_xlabel("X 距离分量 (米)")
    ax.set_ylabel("Y 距离分量 (米)")
    ax.set_zlabel("动态引力奖励值")
    ax.set_title(f"不确定性半径 U_rad = {u_rad_val:.1f}", pad=20)

    # 设置统一的Z轴范围
    ax.set_zlim(0, z_max_overall * 1.05) # 乘以1.05是为了顶部留白

    ax.view_init(elev=30, azim=45)

plt.tight_layout(rect=[0, 0, 1, 0.95]) # 调整布局，为总标题留空间 rect=[left, bottom, right, top]
plt.suptitle("不同不确定性半径下的动态引力奖励模型 (统一Z轴, C2调整)", fontsize=16) # y参数被rect替代
plt.show()