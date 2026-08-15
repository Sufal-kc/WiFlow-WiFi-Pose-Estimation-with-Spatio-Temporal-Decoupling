import numpy as np
import csv
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio
import os.path
import torch.nn as nn
from torch.autograd import Variable
from evaluation import compute_pck_pckh_15
import time
from torchvision.transforms import Resize
from ChannelTrans import ChannelTransformer
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import random
import os
import torchvision
from torchvision.models import ResNet34_Weights  # Add this import
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import glob
import hdf5storage

# ============================== #
# 要保留的关键点定义
# ============================== #
KEEP_KEYPOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
KEYPOINT_MAPPING = {old_idx: new_idx for new_idx, old_idx in enumerate(KEEP_KEYPOINTS)}


# ============================== #
# 数据集和数据加载器 - 使用你提供的数据加载器
# ============================== #
class PreprocessedCSIKeypointsDataset(Dataset):
    """使用预处理后的CSI数据和PAM格式标签的数据集（优化版本）"""

    def __init__(self, csi_data_dir, pam_label_dir, keypoint_scale=1000.0, transform=None, enable_zero_clean=True):
        self.csi_windows = np.load(os.path.join(csi_data_dir, "csi_windows.npy"))

        window_info = np.load(os.path.join(csi_data_dir, "window_info.npz"))
        self.window_to_file = window_info['window_to_file']
        self.window_to_frame = window_info['window_to_frame']

        file_info = np.load(os.path.join(csi_data_dir, "file_info.npz"), allow_pickle=True)
        self.file_ids = file_info['file_ids']
        self.window_ranges = file_info['window_ranges']

        config = np.load(os.path.join(csi_data_dir, "config.npz"))
        self.window_size = config['window_size']
        self.stride = config['stride']

        self.keypoint_scale = keypoint_scale
        self.transform = transform
        self.pam_label_dir = pam_label_dir
        self.enable_zero_clean = enable_zero_clean

        # 缓存单个PAM帧（而不是整个序列）
        self._pam_cache = {}
        self._cache_size = 100  # 增加缓存大小

        print(f"加载了 {len(self.csi_windows)} 个CSI窗口")
        print(f"PAM标签目录: {pam_label_dir}")
        print(f"文件ID数量: {len(self.file_ids)}")
        print(f"零值清理: {'启用(单帧均值法)' if enable_zero_clean else '禁用'}")

    def _get_pam_file_path(self, file_idx, frame_idx):
        """根据文件索引和帧索引获取PAM文件路径"""
        file_id = self.file_ids[file_idx]
        pam_filename = f"{file_id}_dual_cropped_frame_{frame_idx:06d}.mat"

        for person_idx in range(1, 6):
            person_dir = os.path.join(self.pam_label_dir, f"wisppn_labels{person_idx}")
            pam_path = os.path.join(person_dir, pam_filename)
            if os.path.exists(pam_path):
                return pam_path
        return None

    def _clean_single_frame_zeros(self, keypoint):
        """清理单帧中的零值关键点（单帧均值法）"""
        cleaned = keypoint.copy()

        # 找到非零关键点的掩码
        non_zero_mask = (keypoint[:, 0] != 0) | (keypoint[:, 1] != 0)

        # 只有当存在至少一个非零点时才进行处理
        if non_zero_mask.any():
            # 计算非零关键点的平均位置
            mean_pos = keypoint[non_zero_mask].mean(axis=0)
            # 找到所有零值点的索引
            zero_indices = np.where(~non_zero_mask)[0]

            # 用均值替换零值点
            for idx in zero_indices:
                cleaned[idx] = mean_pos

        return cleaned

    def _load_pam_frame(self, file_idx, frame_idx):
        """加载特定帧的PAM数据（按需加载）"""
        cache_key = f"{file_idx}_{frame_idx}"

        if cache_key in self._pam_cache:
            return self._pam_cache[cache_key]

        # 缓存管理 - LRU策略
        if len(self._pam_cache) >= self._cache_size:
            # 删除最旧的缓存项
            oldest_key = next(iter(self._pam_cache))
            del self._pam_cache[oldest_key]

        # 获取PAM文件路径
        pam_path = self._get_pam_file_path(file_idx, frame_idx)

        if pam_path is None:
            # 文件不存在，返回零矩阵
            pam_data = np.zeros((3, 15, 15), dtype=np.float32)
        else:
            try:
                # 加载.mat文件
                data = hdf5storage.loadmat(pam_path)
                pam_data = data['jointsMatrix'][:3, :, :].astype(np.float32)

                # 如果启用清理，应用单帧均值清理
                if self.enable_zero_clean:
                    # 提取关键点坐标
                    keypoints = np.zeros((15, 2), dtype=np.float32)
                    for kp_idx in range(15):
                        keypoints[kp_idx, 0] = pam_data[0, kp_idx, kp_idx]
                        keypoints[kp_idx, 1] = pam_data[1, kp_idx, kp_idx]

                    # 清理零值
                    cleaned_keypoints = self._clean_single_frame_zeros(keypoints)

                    # 更新PAM矩阵
                    for kp_idx in range(15):
                        # 更新对角线（绝对坐标）
                        pam_data[0, kp_idx, kp_idx] = cleaned_keypoints[kp_idx, 0]
                        pam_data[1, kp_idx, kp_idx] = cleaned_keypoints[kp_idx, 1]

                    # 更新非对角线（相对坐标）
                    for i in range(15):
                        for j in range(15):
                            if i != j:
                                pam_data[0, i, j] = cleaned_keypoints[i, 0] - cleaned_keypoints[j, 0]
                                pam_data[1, i, j] = cleaned_keypoints[i, 1] - cleaned_keypoints[j, 1]

                # 归一化坐标
                pam_data[0:2, :, :] = pam_data[0:2, :, :] / self.keypoint_scale
                # 置信度保持不变

            except Exception as e:
                print(f"加载PAM文件失败 {pam_path}: {e}")
                pam_data = np.zeros((3, 15, 15), dtype=np.float32)

        # 缓存结果
        self._pam_cache[cache_key] = pam_data
        return pam_data

    def __len__(self):
        return len(self.csi_windows)

    def __getitem__(self, idx):
        # 获取CSI窗口
        csi_window = self.csi_windows[idx]

        # 获取对应的文件和帧索引
        file_idx = self.window_to_file[idx]
        frame_idx = self.window_to_frame[idx]

        # 按需加载单个PAM帧
        pam_label = self._load_pam_frame(file_idx, frame_idx)

        # 转换为张量
        csi_tensor = torch.from_numpy(csi_window).float()
        pam_tensor = torch.from_numpy(pam_label).float()

        if self.transform:
            csi_tensor = self.transform(csi_tensor)

        return csi_tensor, pam_tensor

    def get_file_indices(self):
        """获取所有文件索引"""
        return list(range(len(self.file_ids)))

    def get_samples_from_file(self, file_idx):
        """获取指定文件的所有样本索引"""
        start_idx, end_idx = self.window_ranges[file_idx]
        return list(range(start_idx, end_idx))


def create_preprocessed_train_val_test_loaders(dataset, batch_size=64, num_workers=0, random_seed=42):
    """按文件级别划分预处理后的数据集"""
    # 设置随机种子
    random.seed(random_seed)

    # 获取所有文件索引
    file_indices = dataset.get_file_indices()
    total_files = len(file_indices)

    # 随机打乱文件顺序
    random.shuffle(file_indices)

    # 按比例划分文件
    train_ratio, val_ratio = 0.7, 0.15
    train_split = int(np.floor(train_ratio * total_files))
    val_split = int(np.floor(val_ratio * total_files))

    # 获取每个集合的文件索引
    train_file_indices = file_indices[:train_split]
    val_file_indices = file_indices[train_split:train_split + val_split]
    test_file_indices = file_indices[train_split + val_split:]

    # 获取每个集合的样本索引
    train_indices = []
    val_indices = []
    test_indices = []

    for file_idx in train_file_indices:
        train_indices.extend(dataset.get_samples_from_file(file_idx))

    for file_idx in val_file_indices:
        val_indices.extend(dataset.get_samples_from_file(file_idx))

    for file_idx in test_file_indices:
        test_indices.extend(dataset.get_samples_from_file(file_idx))

    print(f"训练集: {len(train_indices)} 样本 (来自 {len(train_file_indices)} 个文件)")
    print(f"验证集: {len(val_indices)} 样本 (来自 {len(val_file_indices)} 个文件)")
    print(f"测试集: {len(test_indices)} 样本 (来自 {len(test_file_indices)} 个文件)")

    # 创建子集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader


# ============================== #
# 原始的posenet模型 - 保持不变，只做输入输出适配
# ============================== #
class posenet(nn.Module):
    """
    WiFi-based Human Pose Estimation Network (WPFormer)

    Input: WiFi CSI data from 6 antenna pairs
    Output: 2D pose landmarks (y coordinates for 15 keypoints)

    Modified Architecture:
    - Processes 6 CSI streams from antenna pairs
    - Uses shared ResNet34 encoder for feature extraction
    - Applies Channel Transformer for feature integration
    - Outputs 2×15 pose coordinates via decoder and average pooling
    """

    def __init__(self):
        super(posenet, self).__init__()

        # Create ResNet34 encoder (shared weights for all 6 antenna pairs)
        # FIX: Use weights parameter instead of deprecated pretrained parameter
        # resnet_raw_model1 = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        # 项目目录下的权重文件路径
        project_weight_path = "./resnet34-b627a593.pth"  # 相对路径
        # 或者使用绝对路径：
        # project_weight_path = "/home/aip-dt-01/下载/DY/resnet34-b627a593.pth"

        try:
            if os.path.exists(project_weight_path):
                print(f"发现项目目录下的权重文件: {project_weight_path}")

                # 创建不带权重的ResNet34模型
                resnet_raw_model1 = torchvision.models.resnet34(weights=None)

                # 手动加载权重
                print("正在加载权重...")
                state_dict = torch.load(project_weight_path, map_location='cpu')
                resnet_raw_model1.load_state_dict(state_dict)
                print("✅ 成功从项目目录加载预训练权重！")

            else:
                print(f"❌ 未找到权重文件: {project_weight_path}")
                print("请确保将 resnet34-b627a593.pth 文件放到项目目录下")
                print("使用随机初始化权重...")
                resnet_raw_model1 = torchvision.models.resnet34(weights=None)

        except Exception as e:
            print(f"❌ 加载权重失败: {e}")
            print("使用随机初始化权重...")
            resnet_raw_model1 = torchvision.models.resnet34(weights=None)

        # Expected feature map sizes at each ResNet layer
        filters = [64, 64, 128, 256, 512]

        # Encoder components from ResNet34 (shared across all 6 antenna pairs)
        # Input: 1-channel CSI data -> 64-channel feature maps
        self.encoder_conv1_p1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=False)
        self.encoder_bn1_p1 = resnet_raw_model1.bn1
        self.encoder_relu_p1 = resnet_raw_model1.relu

        # ResNet34 layers (Block 1-5 in paper's Table I)
        self.encoder_layer1_p1 = resnet_raw_model1.layer1  # Block 1: 64×60×32
        self.encoder_layer2_p1 = resnet_raw_model1.layer2  # Block 2: 128×30×16
        self.encoder_layer3_p1 = resnet_raw_model1.layer3  # Block 3: 256×15×8
        # self.encoder_layer4_p1 = resnet_raw_model1.layer4  # Block 4: 512×15×4

        # Channel Transformer for feature integration
        # Input: 512×15×24 (concatenated features from 6 antenna pairs)
        # Output: 512×15×24 (same size, with attention weights)
        self.tf = ChannelTransformer(vis=False, img_size=[15, 144], channel_num=256, num_layers=1, num_heads=3)

        # Decoder: 512×15×24 -> 2×15×24 (pose coordinates)
        self.decode = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False),  # 2 channels for (x,y) coordinates
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(2)  # For final 2D pose output
        self.bn2 = nn.BatchNorm2d(256)  # For concatenated features before transformer
        self.rl = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass through WPFormer network

        Args:
            x: Input CSI data, shape [batch_size, 540, 20]
               Modified to accept the new data format and convert it internally

        Returns:
            x: Pose landmarks, shape [batch_size, 15, 2] (15 keypoints with x,y coordinates)
            time_sum: Forward pass execution time
        """

        # 输入适配：从 [batch_size, 540, 20] 转换为原始模型需要的格式
        batch_size = x.size(0)

        # 将540拆分成18个30（每个天线对）
        x_splits = torch.chunk(x, 18, dim=1)  # 18个 [batch_size, 30, 20]

        # 创建resize函数 - FIX: Add antialias parameter to suppress warning
        torch_resize = Resize([60, 32], antialias=True)

        # 处理所有6个天线对
        x_resized = []
        for i in range(18):
            x_part = x_splits[i]  # [batch_size, 60, 20]
            # 添加channel维度
            x_part = x_part.unsqueeze(1)  # [batch_size, 1, 60, 20]
            # Resize到期望的尺寸
            x_part = torch_resize(x_part)  # [batch_size, 1, 60, 32]
            x_resized.append(x_part)

        time_start = time.time()

        # 对所有6个天线对进行编码
        encoded_features = []

        for x_input in x_resized:
            # Initial convolution layer
            x_feat = self.encoder_conv1_p1(x_input)  # [batch_size, 64, 60, 32]
            x_feat = self.encoder_bn1_p1(x_feat)
            x_feat = self.encoder_relu_p1(x_feat)

            # ResNet34 layers
            x_feat = self.encoder_layer1_p1(x_feat)  # [batch_size, 64, 60, 32]
            x_feat = self.encoder_layer2_p1(x_feat)  # [batch_size, 128, 30, 16]
            x_feat = self.encoder_layer3_p1(x_feat)  # [batch_size, 256, 15, 8]
            # x_feat = self.encoder_layer4_p1(x_feat)  # [batch_size, 512, 15, 4]

            encoded_features.append(x_feat)

        # Concatenation step
        # Concatenate features from 6 antenna pairs along width dimension
        # Each x_i: [batch_size, 512, 15, 4]
        # Result: [batch_size, 512, 15, 24] (4*6=24 in width dimension)
        x = torch.cat(encoded_features, dim=3)

        # Batch normalization before transformer
        x = self.bn2(x)  # [batch_size, 512, 15, 72]

        # Channel Transformer
        # Input: [batch_size, 512, 15, 24] -> Output: [batch_size, 512, 15, 24]
        x, weight = self.tf(x)

        # Decoder
        # Input: [batch_size, 512, 15, 24] -> Output: [batch_size, 2, 15, 24]
        x = self.decode(x)

        # Average pooling
        # Pool across width dimension: [batch_size, 2, 15, 24] -> [batch_size, 2, 15, 1]
        m = torch.nn.AvgPool2d((1, 144), stride=(1, 1))
        x = m(x).squeeze(dim=3)  # [batch_size, 2, 15]

        # Final batch normalization
        x = self.bn1(x)

        time_end = time.time()
        time_sum = time_end - time_start

        # Transpose to get final pose format
        # [batch_size, 2, 15] -> [batch_size, 15, 2]
        x = torch.transpose(x, 1, 2)

        # 输出适配：只保留前15个关键点
        # x = x[:, :15, :]  # [batch_size, 15, 2]

        return x, time_sum


def weights_init(m):
    """
    Initialize network weights

    Args:
        m: Neural network module to initialize
    """
    if isinstance(m, nn.Conv2d):
        # Xavier normal initialization for convolutional layers
        nn.init.xavier_normal_(m.weight.data)
        # Note: bias initialization is commented out
        # nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        # Constant initialization for 2D batch normalization
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        # Constant initialization for 1D batch normalization
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# 在评估函数之前添加：
def mean_keypoint_error(pred, target, keypoint_scale=1.0):
    """
    计算平均关键点误差 (Mean Pose Error)

    Args:
        pred: 预测的关键点 [batch_size, 15, 2]
        target: 真实的关键点 [batch_size, 15, 2]
        keypoint_scale: 缩放因子

    Returns:
        mpe: 平均关键点误差
    """
    # 确保输入维度正确
    if len(pred.shape) == 3 and pred.shape[1] == 2 and pred.shape[2] == 15:
        pred = pred.transpose(1, 2)  # [batch_size, 15, 2]
    if len(target.shape) == 3 and target.shape[1] == 2 and target.shape[2] == 15:
        target = target.transpose(1, 2)  # [batch_size, 15, 2]

    # 计算欧氏距离
    distances = torch.sqrt(torch.sum((pred - target) ** 2, dim=2))  # [batch_size, 15]

    # 计算平均误差
    mpe = torch.mean(distances) * keypoint_scale

    return mpe.item()

def extract_keypoints_from_pam(pam_data, num_keypoints=15):
    """
    从PAM矩阵中提取关键点坐标和置信度（用于测试）
    输入: pam_data [batch_size, 3, 15, 15] - PAM的x'、y'和置信度通道
    输出:
        keypoints [batch_size, num_keypoints, 2] - 提取的关键点坐标
        confidence [batch_size, num_keypoints, 1] - 提取的置信度
    """
    batch_size = pam_data.shape[0]

    # 初始化关键点坐标和置信度
    keypoints = torch.zeros(batch_size, num_keypoints, 2, device=pam_data.device)
    confidence = torch.zeros(batch_size, num_keypoints, 1, device=pam_data.device)

    # 从对角线元素提取坐标和置信度
    for b in range(batch_size):
        for k in range(num_keypoints):
            keypoints[b, k, 0] = pam_data[b, 0, k, k]  # x坐标从第0个通道的对角线
            keypoints[b, k, 1] = pam_data[b, 1, k, k]  # y坐标从第1个通道的对角线
            confidence[b, k, 0] = pam_data[b, 2, k, k]  # 置信度从第2个通道的对角线

    return keypoints, confidence


# ============================== #
# 姿态可视化相关设置
# ============================== #
SKELETON_CONNECTIONS = [
    (0, 1), (1, 8),
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (8, 9), (8, 12),
    (9, 10), (10, 11),
    (12, 13), (13, 14)
]

BODY_PART_COLORS = {
    'head': 'magenta',
    'torso': 'red',
    'left_arm': 'orange',
    'right_arm': 'green',
    'left_leg': 'cyan',
    'right_leg': 'blue'
}

KEYPOINT_GROUPS = {
    'head': [0],
    'torso': [0, 1, 8],
    'left_arm': [2, 3, 4],
    'right_arm': [5, 6, 7],
    'left_leg': [9, 10, 11],
    'right_leg': [12, 13, 14]
}

CONNECTION_COLORS = {
    (0, 1): BODY_PART_COLORS['torso'], (1, 8): BODY_PART_COLORS['torso'],
    (1, 2): BODY_PART_COLORS['left_arm'], (2, 3): BODY_PART_COLORS['left_arm'], (3, 4): BODY_PART_COLORS['left_arm'],
    (1, 5): BODY_PART_COLORS['right_arm'], (5, 6): BODY_PART_COLORS['right_arm'], (6, 7): BODY_PART_COLORS['right_arm'],
    (8, 9): BODY_PART_COLORS['left_leg'], (8, 12): BODY_PART_COLORS['right_leg'],
    (9, 10): BODY_PART_COLORS['left_leg'], (10, 11): BODY_PART_COLORS['left_leg'],
    (12, 13): BODY_PART_COLORS['right_leg'], (13, 14): BODY_PART_COLORS['right_leg']
}


def create_pose_animation_opencv(all_keypoints, output_file="pose_animation.mp4", fps=30,
                                 figsize=(800, 960), keypoint_scale=1000.0):
    """使用OpenCV创建人体姿态动画"""
    width, height = figsize
    frames = len(all_keypoints)

    # 确保关键点是numpy数组
    if torch.is_tensor(all_keypoints):
        all_keypoints = all_keypoints.cpu().numpy()

    # 缩放关键点
    all_keypoints = all_keypoints * keypoint_scale

    # 计算全局边界
    all_x = all_keypoints[:, :, 0].flatten()
    all_y = all_keypoints[:, :, 1].flatten()

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    margin = 0.1
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"生成姿态动画: {output_file}，共 {frames} 帧")

    with tqdm(total=frames, desc="生成视频", unit="帧") as pbar:
        for frame_idx in range(frames):
            # 创建白色背景
            frame_img = np.ones((height, width, 3), dtype=np.uint8) * 255

            # 创建matplotlib图
            fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

            keypoints = all_keypoints[frame_idx]

            # 绘制骨骼连接
            for connection in SKELETON_CONNECTIONS:
                start_idx, end_idx = connection
                color = CONNECTION_COLORS.get(connection, 'gray')
                ax.plot([keypoints[start_idx, 0], keypoints[end_idx, 0]],
                        [keypoints[start_idx, 1], keypoints[end_idx, 1]],
                        color=color, linewidth=3)

            # 绘制关键点
            for part_name, indices in KEYPOINT_GROUPS.items():
                color = BODY_PART_COLORS[part_name]
                part_keypoints = keypoints[indices]
                ax.scatter(part_keypoints[:, 0], part_keypoints[:, 1],
                           c=color, s=50, edgecolors='black')

            # 设置坐标轴
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_max + y_margin, y_min - y_margin)
            ax.set_title(f"Frame {frame_idx + 1}/{frames}", fontsize=14)
            ax.set_aspect('equal')
            ax.axis('off')

            plt.tight_layout()

            # 转换为OpenCV格式
            fig.canvas.draw()
            mat_img = np.array(fig.canvas.renderer.buffer_rgba())
            mat_img = cv2.cvtColor(mat_img, cv2.COLOR_RGBA2BGR)

            # 写入视频
            video_writer.write(mat_img)
            plt.close(fig)
            pbar.update(1)

    video_writer.release()
    print(f"视频生成完成: {output_file}")
    return output_file


def create_side_by_side_video_opencv(true_keypoints, pred_keypoints, output_file="comparison.mp4",
                                     keypoint_scale=1000.0, fps=30):
    """创建对比视频"""
    frames = min(len(true_keypoints), len(pred_keypoints))

    # 转换为numpy数组
    if torch.is_tensor(true_keypoints):
        true_keypoints = true_keypoints.cpu().numpy()
    if torch.is_tensor(pred_keypoints):
        pred_keypoints = pred_keypoints.cpu().numpy()

    # 缩放关键点
    true_keypoints = true_keypoints[:frames] * keypoint_scale
    pred_keypoints = pred_keypoints[:frames] * keypoint_scale

    # 计算全局范围
    all_x = np.concatenate([true_keypoints[:, :, 0].flatten(), pred_keypoints[:, :, 0].flatten()])
    all_y = np.concatenate([true_keypoints[:, :, 1].flatten(), pred_keypoints[:, :, 1].flatten()])

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    margin = 0.1
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin

    width, height = 1600, 800
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"生成对比视频: {output_file}，共 {frames} 帧")

    with tqdm(total=frames, desc="生成对比视频", unit="帧") as pbar:
        for frame_idx in range(frames):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # 真实姿态
            true_kp = true_keypoints[frame_idx]
            for connection in SKELETON_CONNECTIONS:
                start_idx, end_idx = connection
                color = CONNECTION_COLORS.get(connection, 'gray')
                ax1.plot([true_kp[start_idx, 0], true_kp[end_idx, 0]],
                         [true_kp[start_idx, 1], true_kp[end_idx, 1]],
                         color=color, linewidth=3)

            for part_name, indices in KEYPOINT_GROUPS.items():
                color = BODY_PART_COLORS[part_name]
                part_keypoints = true_kp[indices]
                ax1.scatter(part_keypoints[:, 0], part_keypoints[:, 1],
                            c=color, s=50, edgecolors='black')

            ax1.set_xlim(x_min - x_margin, x_max + x_margin)
            ax1.set_ylim(y_max + y_margin, y_min - y_margin)
            ax1.set_title(f"True Pose - Frame {frame_idx + 1}", fontsize=14)
            ax1.set_aspect('equal')
            ax1.axis('off')

            # 预测姿态
            pred_kp = pred_keypoints[frame_idx]
            for connection in SKELETON_CONNECTIONS:
                start_idx, end_idx = connection
                color = CONNECTION_COLORS.get(connection, 'gray')
                ax2.plot([pred_kp[start_idx, 0], pred_kp[end_idx, 0]],
                         [pred_kp[start_idx, 1], pred_kp[end_idx, 1]],
                         color=color, linewidth=3)

            for part_name, indices in KEYPOINT_GROUPS.items():
                color = BODY_PART_COLORS[part_name]
                part_keypoints = pred_kp[indices]
                ax2.scatter(part_keypoints[:, 0], part_keypoints[:, 1],
                            c=color, s=50, edgecolors='black')

            ax2.set_xlim(x_min - x_margin, x_max + x_margin)
            ax2.set_ylim(y_max + y_margin, y_min - y_margin)
            ax2.set_title(f"Predicted Pose - Frame {frame_idx + 1}", fontsize=14)
            ax2.set_aspect('equal')
            ax2.axis('off')

            plt.tight_layout()

            # 转换为OpenCV格式
            fig.canvas.draw()
            mat_img = np.array(fig.canvas.renderer.buffer_rgba())
            mat_img = cv2.cvtColor(mat_img, cv2.COLOR_RGBA2BGR)

            video_writer.write(mat_img)
            plt.close(fig)
            pbar.update(1)

    video_writer.release()
    print(f"对比视频生成完成: {output_file}")
    return output_file


# ============================== #
# 训练函数
# ============================== #
def get_gpu_memory_map():
    """获取所有GPU的显存信息"""
    result = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3  # 转换为GB
            result[i] = gpu_memory
    return result


def calculate_optimal_batch_size(gpu_id):
    """根据GPU显存计算最优批量大小"""
    if not torch.cuda.is_available():
        return 32

    # 获取GPU显存（GB）
    gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 ** 3

    # 根据显存大小设置批量大小
    # 经验公式：每GB显存大约可以处理64个样本
    if gpu_memory > 40:  # 4090 (49GB)
        return 128  # 更大的批量
    elif gpu_memory > 20:  # 2080Ti (22GB)
        return 128
    elif gpu_memory > 10:
        return 32
    else:
        return 32


def setup_distributed(gpu_ids):
    """设置分布式训练环境"""
    if len(gpu_ids) > 1:
        # 初始化分布式训练
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', world_size=len(gpu_ids), rank=0)
        return True
    return False


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_posenet():
    # ============================== #
    # 统一输出目录配置
    # ============================== #
    # 预处理数据目录
    csi_data_dir = "preprocessed_csi_data"
    pam_label_dir = "keypoints_pam_data"  # PAM格式标签目录

    # 统一输出目录配置
    output_dir = "metafi_fix2"  # 主输出目录
    model_dir = os.path.join(output_dir, "models")  # 模型权重目录
    video_dir = os.path.join(output_dir, "videos")  # 视频输出目录
    log_dir = os.path.join(output_dir, "logs")  # 日志目录
    result_dir = os.path.join(output_dir, "results")  # 结果文件目录

    # 创建输出目录结构
    directories = [output_dir, model_dir, video_dir, log_dir, result_dir]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"输出目录已创建: {directory}")

    # 文件路径配置
    best_model_path = os.path.join(model_dir, 'posenet_best.pth')
    latest_model_path = os.path.join(model_dir, 'posenet_latest.pth')

    true_poses_video = os.path.join(video_dir, 'posenet_true_poses.mp4')
    predicted_poses_video = os.path.join(video_dir, 'posenet_predicted_poses.mp4')
    comparison_video = os.path.join(video_dir, 'posenet_comparison.mp4')

    test_results_file = os.path.join(result_dir, 'test_results.json')
    training_curves_file = os.path.join(result_dir, 'training_curves.png')

    print(f"输出目录结构已创建完成: {output_dir}")

    # ============================== #
    # 设备配置
    # ============================== #
    gpu_config = '0'  # 或者你想使用的GPU配置
    if gpu_config == 'auto':
        gpu_memory_map = get_gpu_memory_map()
        rtx4090_ids = [i for i, mem in gpu_memory_map.items() if mem > 40]

        if rtx4090_ids:
            gpu_ids = rtx4090_ids[:1]
            print(f"自动选择: 使用RTX 4090 (GPU {gpu_ids[0]})")
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
            print(f"自动选择: 使用所有GPU {gpu_ids}")
    else:
        gpu_ids = [int(x) for x in gpu_config.split(',')]

    print(f"使用GPU: {gpu_ids}")

    # 打印每个GPU的信息
    for gpu_id in gpu_ids:
        props = torch.cuda.get_device_properties(gpu_id)
        print(f"  GPU {gpu_id}: {props.name}, 显存: {props.total_memory / 1024 ** 3:.1f}GB")

    # 设置主设备
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # 是否使用分布式训练
    use_ddp = False  # 强制使用DataParallel
    if use_ddp:
        setup_distributed(gpu_ids)

    print(f"Using device: {device}")

    # ============================== #
    # 数据加载
    # ============================== #
    keypoint_scale = 1000.0

    # 批量大小
    batch_size = 32

    # 检查数据是否存在
    if not os.path.exists(csi_data_dir) or not os.path.exists(os.path.join(csi_data_dir, "csi_windows.npy")):
        print(f"错误: 未找到CSI预处理数据 {csi_data_dir}")
        return

    if not os.path.exists(pam_label_dir):
        print(f"错误: 未找到PAM标签目录 {pam_label_dir}")
        return

    # 创建数据集
    print("正在加载数据...")
    full_dataset = PreprocessedCSIKeypointsDataset(
        csi_data_dir=csi_data_dir,
        pam_label_dir=pam_label_dir,
        keypoint_scale=1000.0,  # 添加这一行
        enable_zero_clean=True  # 启用零值清理
    )

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_preprocessed_train_val_test_loaders(
        dataset=full_dataset,
        batch_size=batch_size,
        num_workers=0
    )

    # ============================== #
    # 模型初始化
    # ============================== #
    model = posenet()
    model.apply(weights_init)
    model = model.to(device)

    # ========== 极简模型统计（只要这几行） ==========
    # 1. 计算总参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 模型总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 2. 计算FLOPs
    try:
        from thop import profile
        import copy
        model_copy = copy.deepcopy(model)
        input_tensor = torch.randn(1, 540, 20).to(device)
        with torch.no_grad():
            flops, _ = profile(model_copy, inputs=(input_tensor,), verbose=False)
        print(f"💻 模型计算量: {flops / 1e6:.2f}M FLOPs")
        del model_copy
    except:
        print("💻 FLOPs计算需要安装: pip install thop")

    # 设置并行训练
    if use_ddp:
        model = DDP(model, device_ids=gpu_ids, output_device=gpu_ids[0])
        print("使用DistributedDataParallel")
    elif len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print("使用DataParallel")

    # 损失函数和优化器
    criterion_L2 = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 学习率调度器
    n_epochs = 20
    n_epochs_decay = 30
    epoch_count = 1

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
        return lr_l

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    # ============================== #
    # 训练循环
    # ============================== #
    num_epochs = 50
    pck_50_overall_max = 0
    train_mean_loss_iter = []

    print("Starting training...")

    for epoch_index in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss_iter = []

        # 添加训练进度条
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f'Epoch {epoch_index + 1}/{num_epochs} [Train]',
                          unit='batch')

        for idx, (csi_data, pam_batch) in train_pbar:
            csi_data = csi_data.to(device)
            pam_label = pam_batch.to(device)

            # 前向传播
            pred_xy_keypoint, time_forward = model(csi_data)

            # 从PAM标签提取真实关键点
            keypoints, confidence = extract_keypoints_from_pam(pam_label)  # pam_label是完整的[B, 3, 15, 15]

            # 计算损失
            loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, keypoints))

            train_loss_iter.append(loss.cpu().detach().numpy())

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = scheduler.get_last_lr()[0]
            processed_samples = (idx + 1) * batch_size

            # 更新进度条显示信息
            train_pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'lr': f'{lr:.5f}',
                'samples': f'{processed_samples}/{len(train_loader.dataset)}'
            })

        train_pbar.close()

        # 更新学习率
        scheduler.step()

        # 计算平均训练损失
        train_mean_loss = np.mean(train_loss_iter)
        train_mean_loss_iter.append(train_mean_loss)
        print('end of the epoch: %d, with loss: %.3f' % (epoch_index, train_mean_loss))

        # ============================== #
        # 验证阶段
        # ============================== #
        model.eval()
        valid_loss_iter = []
        pck_50_iter = []
        pck_20_iter = []
        mpe_iter = []

        # 添加验证进度条
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch_index + 1}/{num_epochs} [Valid]',
                        unit='batch', leave=False)

        with torch.no_grad():
            for csi_data, pam_batch in val_pbar:
                csi_data = csi_data.to(device)
                pam_label = pam_batch.to(device)

                # 前向传播
                pred_xy_keypoint, time_forward = model(csi_data)

                # 从PAM标签提取真实关键点
                keypoints, confidence = extract_keypoints_from_pam(pam_label)  # pam_label是完整的[B, 3, 15, 15]

                # 计算损失
                loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, keypoints))

                valid_loss_iter.append(loss.cpu().detach().numpy())

                # 计算MPE
                mpe = mean_keypoint_error(pred_xy_keypoint, keypoints, keypoint_scale)
                mpe_iter.append(mpe)

                # 计算PCK - 需要转换格式
                pred_xy_keypoint_pck = pred_xy_keypoint.transpose(1, 2)
                keypoints_pck = keypoints.transpose(1, 2)

                pck_50 = compute_pck_pckh_15(pred_xy_keypoint_pck.cpu(), keypoints_pck.cpu(), 0.5)
                pck_20 = compute_pck_pckh_15(pred_xy_keypoint_pck.cpu(), keypoints_pck.cpu(), 0.2)

                pck_50_iter.append(pck_50)
                pck_20_iter.append(pck_20)

                # 更新验证进度条
                val_pbar.set_postfix({'val_loss': f'{loss:.4f}'})

        val_pbar.close()

        # 计算验证指标
        valid_mean_loss = np.mean(valid_loss_iter)
        valid_mean_mpe = np.mean(mpe_iter)
        pck_50 = np.mean(pck_50_iter, 0)
        pck_20 = np.mean(pck_20_iter, 0)

        pck_50_overall = pck_50[-1] if len(pck_50.shape) > 0 else pck_50
        pck_20_overall = pck_20[-1] if len(pck_20.shape) > 0 else pck_20

        print('validation result with loss: %.3f, mpe: %.3f, pck_50: %.3f, pck_20: %.3f' %
              (valid_mean_loss, valid_mean_mpe, pck_50_overall, pck_20_overall))

        # ============================== #
        # 保存模型 - 使用统一路径配置
        # ============================== #
        if pck_50_overall > pck_50_overall_max:
            print('saving the model at the end of epoch %d with pck_50: %.3f' %
                  (epoch_index, pck_50_overall))
            torch.save(model, best_model_path)
            pck_50_overall_max = pck_50_overall

        # 定期保存最新模型
        if (epoch_index + 1) % 10 == 0:
            torch.save(model, latest_model_path)

        # 定期打印训练历史
        if (epoch_index + 1) % 50 == 0:
            print('the train loss for the first %.1f epoch is' % (epoch_index))
            print(train_mean_loss_iter)

    print("\nTraining completed!")

    # ============================== #
    # 测试阶段
    # ============================== #
    print("\nStarting testing...")

    # 加载最佳模型 - 使用统一路径配置
    model = torch.load(best_model_path, map_location=device)
    model = model.to(device).eval()

    test_loss_iter = []
    pck_50_iter = []
    pck_40_iter = []
    pck_30_iter = []
    pck_20_iter = []
    pck_10_iter = []
    pck_5_iter = []
    mpe_iter = []

    # 保存所有预测和真实关键点用于视频生成
    all_pred_keypoints = []
    all_true_keypoints = []

    # 添加测试进度条
    test_pbar = tqdm(test_loader, desc='Testing', unit='batch')

    with torch.no_grad():
        for csi_data, pam_batch in test_pbar:
            csi_data = csi_data.to(device)
            pam_label = pam_batch.to(device)

            # 前向传播
            pred_xy_keypoint, time_forward = model(csi_data)

            # 从PAM标签提取真实关键点
            keypoints, confidence = extract_keypoints_from_pam(pam_label)  # pam_label是完整的[B, 3, 15, 15]

            # 计算损失
            loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, keypoints))

            test_loss_iter.append(loss.cpu().detach().numpy())

            # 添加MPE计算
            mpe = mean_keypoint_error(pred_xy_keypoint, keypoints, keypoint_scale)
            mpe_iter.append(mpe)

            # 保存预测和真实关键点
            all_pred_keypoints.append(pred_xy_keypoint.cpu().numpy())
            all_true_keypoints.append(keypoints.cpu().numpy())

            # 计算不同阈值的PCK
            pred_xy_keypoint_pck = pred_xy_keypoint.transpose(1, 2)
            keypoints_pck = keypoints.transpose(1, 2)

            pck_50_iter.append(compute_pck_pckh_15(pred_xy_keypoint_pck.cpu(), keypoints_pck.cpu(), 0.5))
            pck_40_iter.append(compute_pck_pckh_15(pred_xy_keypoint_pck.cpu(), keypoints_pck.cpu(), 0.4))
            pck_30_iter.append(compute_pck_pckh_15(pred_xy_keypoint_pck.cpu(), keypoints_pck.cpu(), 0.3))
            pck_20_iter.append(compute_pck_pckh_15(pred_xy_keypoint_pck.cpu(), keypoints_pck.cpu(), 0.2))
            pck_10_iter.append(compute_pck_pckh_15(pred_xy_keypoint_pck.cpu(), keypoints_pck.cpu(), 0.1))
            pck_5_iter.append(compute_pck_pckh_15(pred_xy_keypoint_pck.cpu(), keypoints_pck.cpu(), 0.05))

            # 更新测试进度条
            test_pbar.set_postfix({
                'test_loss': f'{loss:.4f}',
                'time': f'{time_forward:.3f}s'
            })

    test_pbar.close()

    # ============================== #
    # 计算测试结果
    # ============================== #
    test_mean_loss = np.mean(test_loss_iter)
    test_mean_mpe = np.mean(mpe_iter)
    pck_50 = np.mean(pck_50_iter, 0)
    pck_40 = np.mean(pck_40_iter, 0)
    pck_30 = np.mean(pck_30_iter, 0)
    pck_20 = np.mean(pck_20_iter, 0)
    pck_10 = np.mean(pck_10_iter, 0)
    pck_5 = np.mean(pck_5_iter, 0)

    # 获取overall PCK
    pck_50_overall = pck_50[-1] if len(pck_50.shape) > 0 else pck_50
    pck_40_overall = pck_40[-1] if len(pck_40.shape) > 0 else pck_40
    pck_30_overall = pck_30[-1] if len(pck_30.shape) > 0 else pck_30
    pck_20_overall = pck_20[-1] if len(pck_20.shape) > 0 else pck_20
    pck_10_overall = pck_10[-1] if len(pck_10.shape) > 0 else pck_10
    pck_5_overall = pck_5[-1] if len(pck_5.shape) > 0 else pck_5

    print('test result with loss: %.3f, mpe: %.3f, pck_50: %.3f, pck_40: %.3f, pck_30: %.3f, '
          'pck_20: %.3f, pck_10: %.3f, pck_5: %.3f' %
          (test_mean_loss, test_mean_mpe, pck_50_overall, pck_40_overall, pck_30_overall,
           pck_20_overall, pck_10_overall, pck_5_overall))

    print('-----pck_50-----')
    print(pck_50)
    print('-----pck_40-----')
    print(pck_40)
    print('-----pck_30-----')
    print(pck_30)
    print('-----pck_20-----')
    print(pck_20)
    print('-----pck_10-----')
    print(pck_10)
    print('-----pck_5-----')
    print(pck_5)

    # ============================== #
    # 保存测试结果 - 使用统一路径配置
    # ============================== #
    test_results = {
        'test_loss': float(test_mean_loss),
        'test_mpe': float(test_mean_mpe),
        'pck_50': float(pck_50_overall),
        'pck_40': float(pck_40_overall),
        'pck_30': float(pck_30_overall),
        'pck_20': float(pck_20_overall),
        'pck_10': float(pck_10_overall),
        'pck_5': float(pck_5_overall),
        'detailed_pck': {
            'pck_50_per_joint': pck_50.tolist() if hasattr(pck_50, 'tolist') else [float(pck_50)],
            'pck_40_per_joint': pck_40.tolist() if hasattr(pck_40, 'tolist') else [float(pck_40)],
            'pck_30_per_joint': pck_30.tolist() if hasattr(pck_30, 'tolist') else [float(pck_30)],
            'pck_20_per_joint': pck_20.tolist() if hasattr(pck_20, 'tolist') else [float(pck_20)],
            'pck_10_per_joint': pck_10.tolist() if hasattr(pck_10, 'tolist') else [float(pck_10)],
            'pck_5_per_joint': pck_5.tolist() if hasattr(pck_5, 'tolist') else [float(pck_5)]
        },
        'training_config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'keypoint_scale': keypoint_scale,
            'learning_rate': 0.001
        }
    }

    import json
    with open(test_results_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\n测试结果已保存到: {test_results_file}")

    # ============================== #
    # 生成视频 - 使用统一路径配置
    # ============================== #
    if len(all_pred_keypoints) > 0:
        print("\n生成可视化视频...")

        # 转换为numpy数组
        all_pred_keypoints = np.vstack(all_pred_keypoints)
        all_true_keypoints = np.vstack(all_true_keypoints)

        # 选择一部分帧来生成视频（例如前720帧）
        frames_to_animate = min(720, len(all_pred_keypoints))

        # 生成真实姿态视频
        print("生成真实姿态视频...")
        create_pose_animation_opencv(
            all_true_keypoints[:frames_to_animate],
            output_file=true_poses_video,
            keypoint_scale=keypoint_scale
        )

        # 生成预测姿态视频
        print("生成预测姿态视频...")
        create_pose_animation_opencv(
            all_pred_keypoints[:frames_to_animate],
            output_file=predicted_poses_video,
            keypoint_scale=keypoint_scale
        )

        # 生成对比视频
        print("生成对比视频...")
        create_side_by_side_video_opencv(
            all_true_keypoints[:frames_to_animate],
            all_pred_keypoints[:frames_to_animate],
            output_file=comparison_video,
            keypoint_scale=keypoint_scale
        )

        print("\n视频生成完成！")
        print(f"真实姿态视频: {true_poses_video}")
        print(f"预测姿态视频: {predicted_poses_video}")
        print(f"对比视频: {comparison_video}")

    # ============================== #
    # 生成训练曲线图 - 使用统一路径配置
    # ============================== #
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(train_mean_loss_iter, label='Training Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(training_curves_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存到: {training_curves_file}")
    except Exception as e:
        print(f"保存训练曲线失败: {e}")

    print(f"\n所有输出文件已保存到: {output_dir}")
    print("训练完成！")


if __name__ == "__main__":
    train_posenet()