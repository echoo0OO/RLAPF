import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- 1. 地图编码器 (CNN) ---
class MapEncoder(nn.Module):
    """
    使用CNN处理多通道地图输入（奖励地图、无人机位置地图等）。
    """

    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 这一部分需要根据你的输入图像尺寸计算卷积后的平坦化维度
        # 这里我们用一个虚拟输入来动态计算
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 256, 256)
            dummy_output_dim = self.cnn(dummy_input).shape[1]

        self.fc = nn.Linear(dummy_output_dim, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

    def forward(self, map_input):
        # map_input shape: [B, C, H, W]
        conv_out = self.cnn(map_input)
        features = self.fc(conv_out)
        features = self.ln(F.relu(features))
        return features  # shape: [B, feature_dim]


# --- 2. 传感器编码器 (Shared MLP) ---
class SensorEncoder(nn.Module):
    """
    使用共享的MLP为每个传感器生成嵌入向量。
    """

    def __init__(self, sensor_dim, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(sensor_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, sensor_data):
        # sensor_data shape: [B, num_sensors, sensor_dim]
        # MLP会独立地作用在最后一个维度上
        embeddings = self.encoder(sensor_data)
        embeddings = self.ln(F.relu(embeddings))
        return embeddings  # shape: [B, num_sensors, embedding_dim]


# --- 3. 注意力融合主体 (Shared Body) ---
class AttentionFusionBody(nn.Module):
    """
    整合地图编码器、传感器编码器和注意力机制，构成共享的特征提取器。
    """

    def __init__(self, map_channels, sensor_dim, map_feature_dim, sensor_embedding_dim, num_heads=4):
        super().__init__()
        self.map_encoder = MapEncoder(map_channels, map_feature_dim)
        self.sensor_encoder = SensorEncoder(sensor_dim, sensor_embedding_dim)

        # PyTorch的多头注意力模块
        self.attention = nn.MultiheadAttention(
            embed_dim=sensor_embedding_dim,
            num_heads=num_heads,
            batch_first=True  # 让输入/输出的批次维度在第一位，方便处理
        )

        # 最终融合后的全连接层
        self.final_fusion_dim = map_feature_dim + sensor_embedding_dim
        self.fc_out = nn.Linear(self.final_fusion_dim, self.final_fusion_dim)
        self.ln_out = nn.LayerNorm(self.final_fusion_dim)

    def forward(self, map_input, sensor_data):
        # map_input: [B, C, H, W]
        # sensor_data: [B, num_sensors, sensor_dim]

        # 1. 分离编码
        global_context_vector = self.map_encoder(map_input)  # [B, map_feature_dim]
        sensor_embeddings = self.sensor_encoder(sensor_data)  # [B, num_sensors, sensor_embedding_dim]

        # 2. 注意力融合
        # Query: 全局上下文，代表无人机的"提问"
        # Keys/Values: 传感器嵌入，代表可供选择的"答案"
        # 我们需要将全局上下文扩展维度以匹配注意力模块的输入
        query = global_context_vector.unsqueeze(1)  # shape: [B, 1, map_feature_dim]

        # 如果维度不匹配，需要一个线性层来投影
        # 这里我们假设 map_feature_dim == sensor_embedding_dim 以简化
        if global_context_vector.shape[1] != sensor_embeddings.shape[2]:
            raise ValueError("Map feature dim must equal sensor embedding dim for attention.")

        # MultiheadAttention需要Q, K, V
        attended_context_vector, _ = self.attention(
            query=query,
            key=sensor_embeddings,
            value=sensor_embeddings
        )  # output shape: [B, 1, sensor_embedding_dim]

        # 压缩掉中间的序列维度
        attended_context_vector = attended_context_vector.squeeze(1)  # [B, sensor_embedding_dim]

        # 3. 最终融合
        fused_vector = torch.cat([global_context_vector, attended_context_vector], dim=1)
        fused_vector = self.fc_out(fused_vector)
        fused_vector = self.ln_out(F.relu(fused_vector))

        return fused_vector  # [B, final_fusion_dim]


# --- 4. 演员网络 (Actor) ---
class Actor(nn.Module):
    """
    演员网络 = 共享主体 + 演员头
    """

    def __init__(self, body, action_dim, final_fusion_dim):
        super().__init__()
        self.body = body
        self.actor_head = nn.Sequential(
            nn.Linear(final_fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, map_input, sensor_data):
        fused_vector = self.body(map_input, sensor_data)
        # 输出动作，通常会用tanh来限制动作范围在[-1, 1]
        action = torch.tanh(self.actor_head(fused_vector))
        return action


# --- 5. 评论家网络 (Critic) ---
class Critic(nn.Module):
    """
    评论家网络 = 共享主体 + 评论家头
    """

    def __init__(self, body, final_fusion_dim):
        super().__init__()
        self.body = body
        self.critic_head = nn.Sequential(
            nn.Linear(final_fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 输出一个价值
        )

    def forward(self, map_input, sensor_data):
        fused_vector = self.body(map_input, sensor_data)
        value = self.critic_head(fused_vector)
        return value


# --- 使用示例 ---
if __name__ == '__main__':
    # --- 定义超参数 ---
    BATCH_SIZE = 4
    MAP_CHANNELS = 2  # 奖励图 + 无人机位置图
    IMG_H, IMG_W = 256, 256
    NUM_SENSORS = 10
    # 每个传感器的描述: [est_x, est_y, uncertainty_radius, data_volume]
    SENSOR_DIM = 4
    ACTION_DIM = 2  # 例如: [速度, 角速度]

    # 特征维度
    MAP_FEATURE_DIM = 128
    SENSOR_EMBEDDING_DIM = 128  # 注意：为了简化，这里让它和map_feature_dim相等

    # --- 创建虚拟输入数据 ---
    dummy_map_input = torch.randn(BATCH_SIZE, MAP_CHANNELS, IMG_H, IMG_W)
    dummy_sensor_data = torch.randn(BATCH_SIZE, NUM_SENSORS, SENSOR_DIM)

    print(f"输入地图尺寸: {dummy_map_input.shape}")
    print(f"输入传感器数据尺寸: {dummy_sensor_data.shape}\n")

    # --- 实例化网络 ---
    # 1. 先创建共享主体
    shared_body = AttentionFusionBody(
        map_channels=MAP_CHANNELS,
        sensor_dim=SENSOR_DIM,
        map_feature_dim=MAP_FEATURE_DIM,
        sensor_embedding_dim=SENSOR_EMBEDDING_DIM
    )

    # 2. 基于共享主体分别创建Actor和Critic
    actor_net = Actor(shared_body, ACTION_DIM, shared_body.final_fusion_dim)
    critic_net = Critic(shared_body, shared_body.final_fusion_dim)

    # --- 测试前向传播 ---
    print("--- 测试 Actor 网络 ---")
    action_output = actor_net(dummy_map_input, dummy_sensor_data)
    print(f"Actor 输出动作尺寸: {action_output.shape}\n")

    print("--- 测试 Critic 网络 ---")
    value_output = critic_net(dummy_map_input, dummy_sensor_data)
    print(f"Critic 输出价值尺寸: {value_output.shape}")