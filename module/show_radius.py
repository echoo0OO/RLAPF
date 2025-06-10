import numpy as np

# --- 1. 参数设置 ---
# 仿真区域大小
AREA_SIZE = 1000
# 锚点数量
NUM_ANCHORS = 10
# 传感器真实位置
SENSOR_POS_TRUE = np.array([500.0, 500.0])
# 噪声模型参数 g0
G0 = 1.125e-5

# 设置随机数种子以便结果可复现
np.random.seed(42)

# --- 2. 生成锚点并测量距离 ---
# 在区域内随机生成 NUM_ANCHORS 个锚点
anchor_positions = np.random.rand(NUM_ANCHORS, 2) * AREA_SIZE

# 计算传感器到每个锚点的真实水平距离
true_distances = np.linalg.norm(anchor_positions - SENSOR_POS_TRUE, axis=1)

# 根据给定的噪声模型生成带噪声的测量值
# 方差 variance = g0 * (distance^2)
variances = G0 * (true_distances ** 2)
std_devs = np.sqrt(variances)
measured_distances = np.random.normal(loc=true_distances, scale=std_devs)

# --- 3. 使用最小二乘法估计位置 ---
# 我们使用线性最小二乘法来求解。
# 方程: (x - xi)^2 + (y - yi)^2 = di^2
# 展开并与其他方程相减，可以得到一个形如 Ax = b 的线性方程组。

# 选择最后一个锚点作为参考点
ref_anchor_pos = anchor_positions[-1]
ref_measured_dist = measured_distances[-1]

# 构建矩阵 A 和向量 b
A = np.zeros((NUM_ANCHORS - 1, 2))
b = np.zeros(NUM_ANCHORS - 1)

for i in range(NUM_ANCHORS - 1):
    xi, yi = anchor_positions[i]
    di = measured_distances[i]

    # 矩阵 A 的元素
    A[i, 0] = 2 * (ref_anchor_pos[0] - xi)
    A[i, 1] = 2 * (ref_anchor_pos[1] - yi)

    # 向量 b 的元素
    b[i] = di**2 - ref_measured_dist**2 - xi**2 - yi**2 + ref_anchor_pos[0]**2 + ref_anchor_pos[1]**2

# 使用伪逆求解 Ax = b
# pos_est = (A^T A)^-1 A^T b
try:
    pos_estimated = np.linalg.pinv(A) @ b
except np.linalg.LinAlgError:
    pos_estimated = np.array([np.nan, np.nan]) # 如果矩阵奇异则无法求解

# --- 4. 计算估计半径和误差 ---
# 计算估计位置与真实位置之间的欧氏距离误差
position_error = np.linalg.norm(pos_estimated - SENSOR_POS_TRUE)

# 计算估计半径（估计点到所有锚点的平均距离）
distances_from_estimated = np.linalg.norm(anchor_positions - pos_estimated, axis=1)
radius_estimated = np.mean(distances_from_estimated)


# --- 5. 打印结果 ---
# (此部分代码用于生成上文的输出结果)