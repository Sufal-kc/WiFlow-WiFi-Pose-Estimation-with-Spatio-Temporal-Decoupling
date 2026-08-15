import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
from tqdm import tqdm
import copy
import random
import re
import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import psutil
import gc
import cv2
import math
from math import sqrt
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy import stats
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import weight_norm
from collections import deque
import warnings
import pickle
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

# ============================== #
# 修复字体设置
# ============================== #
try:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    print("字体设置成功")
except Exception as e:
    print(f"字体设置错误: {e}")

def safe_text(text):
    """如果需要，将中文字符转换为英文等效字符"""
    try:
        translation = {
            '样本': 'Sample',
            '真实姿态': 'True Pose',
            '预测姿态': 'Predicted Pose',
            '帧': 'Frame',
            '身体部位': 'Body Parts',
            '头部': 'Head',
            '躯干': 'Torso',
            '左臂': 'Left Arm',
            '右臂': 'Right Arm',
            '左腿': 'Left Leg',
            '右腿': 'Right Leg',
            '人体姿态': 'Human Pose',
            '中心点/颈部': 'Neck/Center',
            '胸部中心': 'Chest Center',
            '左肩': 'Left Shoulder',
            '右肩': 'Right Shoulder',
            '左肘': 'Left Elbow',
            '右肘': 'Right Elbow',
            '左手腕': 'Left Wrist',
            '右手腕': 'Right Wrist',
            '骨盆': 'Pelvis',
            '左髋': 'Left Hip',
            '右髋': 'Right Hip',
            '左膝': 'Left Knee',
            '右膝': 'Right Knee',
            '左踝': 'Left Ankle',
            '右踝': 'Right Ankle',
            '左颊': 'Left Cheek',
            '右颊': 'Right Cheek',
            '左耳': 'Left Ear',
            '右耳': 'Right Ear',
            '左脚大拇指': 'Left Foot Thumb',
            '右脚大拇指': 'Right Foot Thumb',
            '右脚小拇指': 'Right Foot Pinky',
            '左脚小拇指': 'Left Foot Pinky',
            '左脚跟': 'Left Heel',
            '右脚跟': 'Right Heel',
            '对比视频进度': 'Comparison Video Progress',
            '视频生成进度': 'Video Generation Progress',
            '开始生成视频': 'Starting Video Generation',
            '视频生成完成': 'Video Generation Complete',
            '对比视频生成完成': 'Comparison Video Complete'
        }

        for cn, en in translation.items():
            if cn in text:
                text = text.replace(cn, en)
        return text
    except:
        return text

# ============================== #
# 1. 定义要保留的关键点
# ============================== #
KEEP_KEYPOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
KEYPOINT_MAPPING = {old_idx: new_idx for new_idx, old_idx in enumerate(KEEP_KEYPOINTS)}

# ============================== #
# 全局配置和实用函数
# ============================== #
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================== #
# 数据集和数据加载器
# ============================== #
class PreprocessedCSIKeypointsDataset(Dataset):
    """
    使用预处理后的CSI数据和关键点的数据集
    自动检测并使用NPY格式（如果存在）以提升速度
    """

    def __init__(self, data_dir, keypoint_scale=1000.0, transform=None, enable_temporal_clean=True):
        # 加载CSI窗口数据
        self.csi_windows = np.load(os.path.join(data_dir, "csi_windows.npy"))

        # 加载窗口元数据
        window_info = np.load(os.path.join(data_dir, "window_info.npz"))
        self.window_to_file = window_info['window_to_file']
        self.window_to_frame = window_info['window_to_frame']

        # 加载文件元数据
        file_info = np.load(os.path.join(data_dir, "file_info.npz"), allow_pickle=True)
        self.keypoints_files = file_info['keypoints_files']
        self.file_ids = file_info['file_ids']
        self.window_ranges = file_info['window_ranges']

        # 加载配置
        config = np.load(os.path.join(data_dir, "config.npz"))
        self.window_size = config['window_size']
        self.stride = config['stride']

        self.keypoint_scale = keypoint_scale
        self.transform = transform
        self.enable_temporal_clean = enable_temporal_clean

        # 检测并加载NPY格式的关键点（如果存在）
        self.use_npy_mode = False
        self.all_keypoints = None
        self.file_mappings = None

        # 检查是否存在处理好的NPY数据
        all_keypoints_path = os.path.join(data_dir, "all_keypoints.npy")
        file_mappings_path = os.path.join(data_dir, "file_mappings.pkl")

        if os.path.exists(all_keypoints_path) and os.path.exists(file_mappings_path):
            print("检测到NPY格式关键点，使用快速加载模式...")
            self.all_keypoints = np.load(all_keypoints_path)
            with open(file_mappings_path, 'rb') as f:
                self.file_mappings = pickle.load(f)
            self.use_npy_mode = True
            print(f"加载了 {len(self.csi_windows)} 个CSI窗口（NPY快速模式）")
            print(f"关键点数据形状: {self.all_keypoints.shape}")
        else:
            # 回退到CSV模式
            print("未检测到NPY格式关键点，使用CSV模式（较慢）...")
            print("建议运行 preprocess_keypoints_to_npy.py 以加速训练")

            # 缓存清理后的关键点序列
            self._cleaned_keypoints_cache = {}
            self._raw_keypoints_cache = {}
            self._cache_size = 10

            print(f"加载了 {len(self.csi_windows)} 个CSI窗口，来自 {len(self.keypoints_files)} 个文件")

        print(f"零值清理: {'启用' if enable_temporal_clean else '禁用'}")

    def __len__(self):
        return len(self.csi_windows)

    def _get_keypoint_npy(self, idx):
        """NPY模式：从预处理的NPY数据获取关键点"""
        file_idx = self.window_to_file[idx]
        frame_idx = self.window_to_frame[idx]

        # 获取对应的CSV文件路径
        csv_file = self.keypoints_files[file_idx]

        # 从映射中获取数据索引
        if csv_file in self.file_mappings:
            mapping = self.file_mappings[csv_file]
            global_frame_idx = mapping['start_idx'] + frame_idx

            if global_frame_idx < len(self.all_keypoints):
                keypoint = self.all_keypoints[global_frame_idx]

                # 如果需要清理零值
                if self.enable_temporal_clean:
                    keypoint = self._clean_single_frame_zeros(keypoint)

                return keypoint

        # 如果找不到，返回零
        return np.zeros((15, 2), dtype=np.float32)

    def _clean_single_frame_zeros(self, keypoint):
        """清理单帧中的零值关键点（简单处理）"""
        cleaned = keypoint.copy()

        # 找到非零关键点的平均位置
        non_zero_mask = (keypoint[:, 0] != 0) | (keypoint[:, 1] != 0)

        if non_zero_mask.any():
            # 用非零关键点的平均值替代零值
            mean_pos = keypoint[non_zero_mask].mean(axis=0)
            zero_indices = np.where(~non_zero_mask)[0]

            for idx in zero_indices:
                cleaned[idx] = mean_pos

        return cleaned

    def _load_raw_keypoints(self, file_idx):
        """CSV模式：加载原始关键点序列"""
        if file_idx in self._raw_keypoints_cache:
            return self._raw_keypoints_cache[file_idx]

        # 缓存管理
        if len(self._raw_keypoints_cache) >= self._cache_size:
            oldest_key = next(iter(self._raw_keypoints_cache))
            del self._raw_keypoints_cache[oldest_key]
            if oldest_key in self._cleaned_keypoints_cache:
                del self._cleaned_keypoints_cache[oldest_key]

        keypoints_file = self.keypoints_files[file_idx]

        # 读取关键点数据
        keypoints_data = pd.read_csv(keypoints_file, header=0).values

        # 处理数据格式
        if keypoints_data.shape[1] > 50:
            keypoints_data = keypoints_data[:, -50:]

        # 归一化
        keypoints_data = keypoints_data.astype(np.float32) / self.keypoint_scale

        # 重塑为 (num_frames, 25, 2)
        num_frames = len(keypoints_data)
        keypoints_reshaped = keypoints_data.reshape(num_frames, 25, 2)

        # 只保留需要的15个关键点
        filtered_keypoints = keypoints_reshaped[:, KEEP_KEYPOINTS, :]

        # 缓存
        self._raw_keypoints_cache[file_idx] = filtered_keypoints

        return filtered_keypoints

    def _clean_zero_keypoints(self, keypoints_sequence):
        """CSV模式：只处理零值异常点，通过插值修复"""
        num_frames, num_keypoints, _ = keypoints_sequence.shape
        coords = keypoints_sequence.copy()

        for kp_idx in range(num_keypoints):
            # 找到所有零值点
            zero_indices = []
            for t in range(num_frames):
                if coords[t, kp_idx, 0] == 0 and coords[t, kp_idx, 1] == 0:
                    zero_indices.append(t)

            # 修复零值点
            for t in zero_indices:
                # 找到前后最近的非零点
                valid_prev = None
                valid_next = None

                # 向前搜索非零点
                for prev_t in range(t - 1, -1, -1):
                    if not (coords[prev_t, kp_idx, 0] == 0 and coords[prev_t, kp_idx, 1] == 0):
                        valid_prev = prev_t
                        break

                # 向后搜索非零点
                for next_t in range(t + 1, num_frames):
                    if not (coords[next_t, kp_idx, 0] == 0 and coords[next_t, kp_idx, 1] == 0):
                        valid_next = next_t
                        break

                # 修复策略
                if valid_prev is not None and valid_next is not None:
                    # 线性插值
                    alpha = (t - valid_prev) / (valid_next - valid_prev)
                    coords[t, kp_idx] = (1 - alpha) * coords[valid_prev, kp_idx] + \
                                        alpha * coords[valid_next, kp_idx]
                elif valid_prev is not None:
                    # 使用前一个有效点
                    coords[t, kp_idx] = coords[valid_prev, kp_idx]
                elif valid_next is not None:
                    # 使用后一个有效点
                    coords[t, kp_idx] = coords[valid_next, kp_idx]

        return coords

    def _load_and_clean_keypoints(self, file_idx):
        """CSV模式：加载并清理关键点序列"""
        if not self.enable_temporal_clean:
            return self._load_raw_keypoints(file_idx)

        if file_idx in self._cleaned_keypoints_cache:
            return self._cleaned_keypoints_cache[file_idx]

        raw_keypoints = self._load_raw_keypoints(file_idx)
        cleaned_keypoints = self._clean_zero_keypoints(raw_keypoints)
        self._cleaned_keypoints_cache[file_idx] = cleaned_keypoints

        return cleaned_keypoints

    def __getitem__(self, idx):
        # 获取CSI窗口
        csi_window = self.csi_windows[idx]

        # 根据模式获取关键点
        if self.use_npy_mode:
            # NPY快速模式
            keypoint = self._get_keypoint_npy(idx)
        else:
            # CSV模式
            file_idx = self.window_to_file[idx]
            frame_idx = self.window_to_frame[idx]

            # 获取清理后的关键点序列
            keypoints_sequence = self._load_and_clean_keypoints(file_idx)

            # 提取对应帧的关键点
            keypoint = keypoints_sequence[frame_idx]

        # 转换为张量
        csi_tensor = torch.from_numpy(csi_window).float()
        keypoint_tensor = torch.from_numpy(keypoint).float()

        # 应用变换
        if self.transform:
            csi_tensor = self.transform(csi_tensor)

        return csi_tensor, keypoint_tensor

    def get_file_indices(self):
        """获取所有文件索引"""
        return list(range(len(self.keypoints_files)))

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

def check_data_ranges(data_loader):
    """检查CSI和关键点的值范围"""
    csi_min, csi_max = float('inf'), -float('inf')
    kpt_min, kpt_max = float('inf'), -float('inf')

    # 检查一部分批次
    for i, (csi, keypoints) in enumerate(data_loader):
        csi_min = min(csi_min, csi.min().item())
        csi_max = max(csi_max, csi.max().item())
        kpt_min = min(kpt_min, keypoints.min().item())
        kpt_max = max(kpt_max, keypoints.max().item())

        # 只检查前5个批次
        if i >= 5:
            break

    print(f"CSI数据范围: [{csi_min:.4f}, {csi_max:.4f}]")
    print(f"关键点数据范围: [{kpt_min:.4f}, {kpt_max:.4f}]")
    return csi_min, csi_max, kpt_min, kpt_max

# ============================== #
# 模型组件
# ============================== #
class SKConv(nn.Module):
    """修复了张量维度错位bug的稳健版 SKConv"""

    def __init__(self, input_dim, output_dim, dim1, dim2, M=4, G=1, r=4, stride=1):
        super(SKConv, self).__init__()
        self.M = M
        self.output_dim = output_dim

        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride,
                          padding=1 + i, dilation=1 + i, groups=G, bias=False),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True)
            ))

        # SKNet的通道衰减设计，保证最小维度为32
        d = max(int(output_dim / r), 32)

        # 针对 freq-chan 的注意力全连接层
        self.fc = nn.Sequential(
            nn.Conv1d(output_dim, d, kernel_size=1, stride=1),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True)
        )

        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Conv1d(d, output_dim, kernel_size=1, stride=1))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 1. Split & Transform
        feats = [conv(x) for conv in self.convs]
        feats = torch.stack(feats, dim=1)  # [B, M, C, H, W]

        # 2. Fuse (将不同分支的特征相加)
        feats_U = torch.sum(feats, dim=1)  # [B, C, H, W]

        # 沿着时间维度 (W) 求平均，保留频率维度 (H) -> [B, C, H]
        feats_S = torch.mean(feats_U, dim=3)

        # 3. Select (注意力生成)
        feats_Z = self.fc(feats_S)  # [B, d, H]

        attention_vectors = [fc(feats_Z) for fc in self.fcs]  # list of M [B, C, H]
        attention_vectors = torch.stack(attention_vectors, dim=1)  # [B, M, C, H]

        # 在 M (分支) 维度上进行 Softmax
        attention_vectors = self.softmax(attention_vectors)  # [B, M, C, H]

        # 扩展出 W 维度用于广播相乘
        attention_vectors = attention_vectors.unsqueeze(-1)  # [B, M, C, H, 1]

        # 4. Multiply and Sum (特征按注意力权重加权融合)
        feats_V = torch.sum(feats * attention_vectors, dim=1)  # [B, C, H, W]

        return feats_V


class SKUnit(nn.Module):
    def __init__(self, in_features, mid_features, out_features, dim1, dim2, M=4, G=1, r=4, stride=1):
        super(SKUnit, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=stride, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
        )

        self.conv2_sk = nn.Sequential(
            SKConv(input_dim=mid_features, output_dim=out_features, dim1=dim1, dim2=dim2, M=M, G=G, r=r, stride=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_sk(out)
        return out


class HPELINet_Baseline(nn.Module):
    """
    单任务版本的 HPELINet，完美适配 new7.py
    输入: [B, 540, 20]
    输出: [B, 15, 2]
    """

    def __init__(self):
        super(HPELINet_Baseline, self).__init__()

        # --- CNN-spatio 及 Selective Kernel 模块 ---
        # 第一层: 输入3通道，升维到64
        self.skunit1 = SKUnit(in_features=3, mid_features=64, out_features=64,
                              dim1=180, dim2=20, M=4, G=1, r=4)

        self.pool1 = nn.AvgPool2d((2, 2))  # [B, 64, 180, 20] -> [B, 64, 90, 10]

        # 第二层: 输入64通道，升维到128
        self.skunit2 = SKUnit(in_features=64, mid_features=128, out_features=128,
                              dim1=90, dim2=10, M=4, G=1, r=4)

        self.pool2 = nn.AvgPool2d((2, 2))  # [B, 128, 90, 10] -> [B, 128, 45, 5]

        # --- Regression 回归模块 (原论文参数重构) ---
        self.regression = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 1), stride=(2, 1), padding=0),  # -> [B, 64, 22, 5]
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=(3, 1), stride=(2, 1), padding=0),  # -> [B, 32, 10, 5]
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, kernel_size=(3, 1), stride=(1, 1), padding=0),  # -> [B, 16, 8, 5]
            nn.ReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(16 * 8 * 5, 15 * 2)  # 直接映射到 15个关键点，每个点2维坐标
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b = x.shape[0]

        # 1. 魔法物理映射：[B, 540, 20] -> [B, 3, 180, 20]
        x = x.view(b, 3, 180, 20)

        # 2. SKUnit 特征提取与池化
        x = self.skunit1(x)
        x = self.pool1(x)

        x = self.skunit2(x)
        x = self.pool2(x)

        # 3. 姿态回归计算
        x = self.regression(x)

        # 4. 重塑为输出格式 [B, 15, 2]
        x = x.view(b, 15, 2)

        return x

class SimplePoseLoss(nn.Module):
    """简化的姿态损失函数 - 只包含位置和骨骼长度损失"""
    def __init__(self,
                 position_weight=1.0,
                 bone_length_weight=0.2,
                 loss_type='smooth_l1'):  # 'mse', 'l1', 'smooth_l1'
        super().__init__()

        self.position_weight = position_weight
        self.bone_length_weight = bone_length_weight
        self.loss_type = loss_type

        self.bone_connections = [
            (0, 1), (1, 8), (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7), (8, 9), (8, 12),
            (9, 10), (10, 11), (12, 13), (13, 14)
        ]

    def compute_bone_lengths(self, keypoints):
        """计算骨骼长度"""
        bone_lengths = []
        for start_idx, end_idx in self.bone_connections:
            bone_vec = keypoints[..., end_idx, :] - keypoints[..., start_idx, :]
            bone_length = torch.sqrt(torch.sum(bone_vec ** 2, dim=-1) + 1e-8)
            bone_lengths.append(bone_length)
        return torch.stack(bone_lengths, dim=-1)

    def forward(self, pred, target):
        """
        pred: [batch_size, 15, 2] - 预测的关键点
        target: [batch_size, 15, 2] - 真实的关键点
        """
        batch_size = pred.shape[0]

        # 处理形状不匹配
        if pred.shape != target.shape:
            if len(pred.shape) == 2 and pred.shape[1] == 30:
                pred = pred.reshape(batch_size, 15, 2)
            if len(target.shape) == 2 and target.shape[1] == 30:
                target = target.reshape(batch_size, 15, 2)

        # 1. 位置损失 - 使用不同的损失函数
        if self.loss_type == 'mse':
            position_loss = F.mse_loss(pred, target)
        elif self.loss_type == 'l1':
            position_loss = F.l1_loss(pred, target)
        elif self.loss_type == 'smooth_l1':
            position_loss = F.smooth_l1_loss(pred, target, beta=0.1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # 2. 骨骼长度约束损失
        try:
            pred_bone_lengths = self.compute_bone_lengths(pred)
            target_bone_lengths = self.compute_bone_lengths(target)

            if self.loss_type == 'mse':
                bone_loss = F.mse_loss(pred_bone_lengths, target_bone_lengths)
            elif self.loss_type == 'l1':
                bone_loss = F.l1_loss(pred_bone_lengths, target_bone_lengths)
            elif self.loss_type == 'smooth_l1':
                bone_loss = F.smooth_l1_loss(pred_bone_lengths, target_bone_lengths, beta=0.05)

        except Exception as e:
            print(f"骨骼长度计算错误: {e}")
            bone_loss = torch.tensor(0.0, device=pred.device)

        # 损失字典
        loss_dict = {
            'position': position_loss.item(),
            'bone': bone_loss.item()
        }

        # 总损失
        total_loss = (self.position_weight * position_loss +
                      self.bone_length_weight * bone_loss)

        return total_loss, loss_dict


def extract_keypoints_from_pam(pam_data, num_keypoints=15):
    """
    从PAM矩阵中提取关键点坐标（用于测试）
    输入: pam_data [batch_size, 2, 15, 15] - PAM的x'和y'通道
    输出: keypoints [batch_size, num_keypoints, 2] - 提取的关键点坐标
    """
    batch_size = pam_data.shape[0]

    # 初始化关键点坐标
    keypoints = torch.zeros(batch_size, num_keypoints, 2, device=pam_data.device)

    # 从对角线元素提取坐标
    for b in range(batch_size):
        for k in range(num_keypoints):
            keypoints[b, k, 0] = pam_data[b, 0, k, k]  # x坐标从第0个通道的对角线
            keypoints[b, k, 1] = pam_data[b, 1, k, k]  # y坐标从第1个通道的对角线

    return keypoints

# ============================== #
# 评估函数
# ============================== #
def percentage_correct_keypoints(pred, target, thresholds=[0.2], use_torso_norm=True):
    """计算PCK (Percentage of Correct Keypoints) - 支持多个阈值 - 修复版本"""
    batch_size = pred.shape[0]

    # 修复：使用reshape替代view
    if len(pred.shape) == 2 and pred.shape[1] == 30:
        pred = pred.reshape(batch_size, 15, 2)
        target = target.reshape(batch_size, 15, 2)

    if use_torso_norm:
        NECK_IDX = 2
        PELVIS_IDX = 12
        normalize_distances = torch.sqrt(
            torch.sum((target[:, NECK_IDX] - target[:, PELVIS_IDX]) ** 2, dim=1)
        )
    else:
        LEFT_SHOULDER_IDX = 2
        RIGHT_SHOULDER_IDX = 5
        normalize_distances = torch.sqrt(
            torch.sum((target[:, LEFT_SHOULDER_IDX] - target[:, RIGHT_SHOULDER_IDX]) ** 2, dim=1)
        )

    normalize_distances = torch.clamp(normalize_distances, min=0.01)
    distances = torch.sqrt(torch.sum((pred - target) ** 2, dim=2))
    normalized_distances = distances / normalize_distances.unsqueeze(1)

    pck_results = {}
    for threshold in thresholds:
        correct_keypoints = (normalized_distances <= threshold).float()
        pck_overall = correct_keypoints.mean()
        pck_results[threshold] = pck_overall.item()

    first_threshold = thresholds[0]
    return pck_results

def mean_keypoint_error(pred, target):
    """计算平均关键点距离 - 修复版本"""
    batch_size = pred.shape[0]

    # 修复：使用reshape替代view
    if len(pred.shape) == 2 and pred.shape[1] == 30:
        pred = pred.reshape(batch_size, 15, 2)
        target = target.reshape(batch_size, 15, 2)

    distances = torch.sqrt(torch.sum((pred - target) ** 2, dim=2))
    mean_distance = torch.mean(distances)

    return mean_distance.item()


# ============================== #
# 数据增强函数
# ============================== #
def time_masking(x, mask_ratio=0.3, mask_len_range=(5, 10)):
    """时间掩码增强：随机掩码时间段落并用均值填充"""
    device = x.device
    masked_x = x.clone()
    B, C, T = masked_x.shape

    for i in range(B):
        if torch.rand(1).item() < mask_ratio:
            num_masks = torch.randint(1, 3, (1,)).item()
            for _ in range(num_masks):
                mask_len = torch.randint(mask_len_range[0], mask_len_range[1], (1,)).item()
                start = torch.randint(0, T - mask_len, (1,)).item()
                for c in range(C):
                    mean_val = masked_x[i, c, :].mean()
                    masked_x[i, c, start:start + mask_len] = mean_val

    return masked_x

def add_noise(x, noise_level=0.05):
    """添加随机噪声"""
    device = x.device
    noise = torch.randn_like(x).to(device) * noise_level * torch.std(x)
    return x + noise

def random_scaling(x, scale_range=(0.9, 1.1)):
    """随机缩放信号幅度"""
    device = x.device
    if torch.rand(1).item() < 0.5:
        scale_factor = torch.FloatTensor(1).uniform_(scale_range[0], scale_range[1]).to(device)
        return x * scale_factor
    return x

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

KEYPOINT_NAMES = {
    0: "中心点/颈部", 1: "胸部中心", 2: "左肩", 3: "左肘", 4: "左手腕",
    5: "右肩", 6: "右肘", 7: "右手腕", 8: "骨盆", 9: "左髋",
    10: "左膝", 11: "左踝", 12: "右髋", 13: "右膝", 14: "右踝"
}

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

def visualize_pose(keypoints, title="人体姿态", figsize=(10, 12), show_labels=True, show_legend=True):
    """可视化单帧人体姿态"""
    fig, ax = plt.subplots(figsize=figsize)

    for connection in SKELETON_CONNECTIONS:
        start_idx, end_idx = connection
        color = CONNECTION_COLORS.get(connection, 'gray')
        ax.plot([keypoints[start_idx, 0], keypoints[end_idx, 0]],
                [keypoints[start_idx, 1], keypoints[end_idx, 1]],
                color=color, linewidth=3)

    for part_name, indices in KEYPOINT_GROUPS.items():
        color = BODY_PART_COLORS[part_name]
        part_keypoints = keypoints[indices]
        ax.scatter(part_keypoints[:, 0], part_keypoints[:, 1],
                   c=color, s=50, edgecolors='black', label=safe_text(part_name))

    if show_labels:
        for i, (x, y) in enumerate(keypoints):
            ax.text(x, y, str(i), fontsize=10, ha='center', va='center', color='white',
                    bbox=dict(boxstyle="circle,pad=0.1", fc='black', ec='none', alpha=0.7))

    if show_legend:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=safe_text(name))
            for name, color in BODY_PART_COLORS.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right', title=safe_text("身体部位"))

    ax.set_title(safe_text(title), fontsize=14)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    return fig, ax

def create_pose_animation_opencv(all_keypoints, output_file="pose_animation.mp4", fps=30,
                                 figsize=(800, 960), keypoint_scale=1.0,
                                 show_labels=True, show_legend=True):
    """使用OpenCV和tqdm创建人体姿态动画"""
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from tqdm import tqdm

    width, height = figsize
    frames = len(all_keypoints)
    reshaped_keypoints = all_keypoints.reshape(frames, -1, 2)

    if keypoint_scale != 1.0:
        reshaped_keypoints *= keypoint_scale

    all_x = reshaped_keypoints[:, :, 0].flatten()
    all_y = reshaped_keypoints[:, :, 1].flatten()

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    margin = 0.1
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin

    if show_legend:
        legend_fig = plt.figure(figsize=(width / 100, 1))
        legend_ax = legend_fig.add_subplot(111)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=safe_text(name))
            for name, color in BODY_PART_COLORS.items()
        ]
        legend_ax.legend(handles=legend_elements, loc='center', ncol=len(BODY_PART_COLORS), title=safe_text("身体部位"))
        legend_ax.axis('off')

        canvas = FigureCanvas(legend_fig)
        canvas.draw()
        legend_arr = np.array(canvas.renderer.buffer_rgba())
        legend_arr = cv2.cvtColor(legend_arr, cv2.COLOR_RGBA2BGR)
        legend_height = legend_arr.shape[0]
        plt.close(legend_fig)
    else:
        legend_arr = None
        legend_height = 0

    total_height = height + legend_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, total_height))

    print(safe_text(f"开始生成视频: {output_file}，共 {frames} 帧"))

    with tqdm(total=frames, desc="生成视频", unit="帧") as pbar:
        for frame_idx in range(frames):
            frame_img = np.ones((total_height, width, 3), dtype=np.uint8) * 255

            fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

            keypoints = reshaped_keypoints[frame_idx]

            for connection in SKELETON_CONNECTIONS:
                start_idx, end_idx = connection
                color = CONNECTION_COLORS.get(connection, 'gray')
                ax.plot([keypoints[start_idx, 0], keypoints[end_idx, 0]],
                        [keypoints[start_idx, 1], keypoints[end_idx, 1]],
                        color=color, linewidth=3)

            for part_name, indices in KEYPOINT_GROUPS.items():
                color = BODY_PART_COLORS[part_name]
                part_keypoints = keypoints[indices]
                ax.scatter(part_keypoints[:, 0], part_keypoints[:, 1],
                           c=color, s=50, edgecolors='black')

            if show_labels:
                for i, (x, y) in enumerate(keypoints):
                    ax.text(x, y, str(i), fontsize=10, ha='center', va='center', color='white',
                            bbox=dict(boxstyle="circle,pad=0.1", fc='black', ec='none', alpha=0.7))

            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_max + y_margin, y_min - y_margin)
            ax.set_title(safe_text(f"姿态 - 帧 {frame_idx + 1}/{frames}"), fontsize=14)
            ax.set_aspect('equal')
            ax.axis('off')

            plt.tight_layout()

            canvas = FigureCanvas(fig)
            canvas.draw()
            mat_img = np.array(canvas.renderer.buffer_rgba())
            mat_img = cv2.cvtColor(mat_img, cv2.COLOR_RGBA2BGR)

            h, w = mat_img.shape[:2]
            frame_img[:h, :w] = mat_img

            if show_legend and legend_arr is not None:
                lh, lw = legend_arr.shape[:2]
                y_offset = h
                frame_img[y_offset:y_offset + lh, :lw] = legend_arr

            video_writer.write(frame_img)
            plt.close(fig)
            pbar.update(1)

    video_writer.release()
    print(safe_text(f"视频生成完成: {output_file}"))
    return output_file

def create_side_by_side_video_opencv(true_keypoints, pred_keypoints, output_file="comparison.mp4",
                                     keypoint_scale=1.0, fps=30):
    """创建对比视频"""
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from tqdm import tqdm

    frames = min(len(true_keypoints), len(pred_keypoints))
    true_reshaped = true_keypoints[:frames].reshape(frames, -1, 2)
    pred_reshaped = pred_keypoints[:frames].reshape(frames, -1, 2)

    if keypoint_scale != 1.0:
        true_reshaped *= keypoint_scale
        pred_reshaped *= keypoint_scale

    # 计算全局范围
    all_x = np.concatenate([true_reshaped[:, :, 0].flatten(), pred_reshaped[:, :, 0].flatten()])
    all_y = np.concatenate([true_reshaped[:, :, 1].flatten(), pred_reshaped[:, :, 1].flatten()])

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    margin = 0.1
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin

    width, height = 1600, 800
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"开始生成对比视频: {output_file}，共 {frames} 帧")

    with tqdm(total=frames, desc="生成对比视频", unit="帧") as pbar:
        for frame_idx in range(frames):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # 真实姿态
            true_kp = true_reshaped[frame_idx]
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
            pred_kp = pred_reshaped[frame_idx]
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

            canvas = FigureCanvas(fig)
            canvas.draw()
            mat_img = np.array(canvas.renderer.buffer_rgba())
            mat_img = cv2.cvtColor(mat_img, cv2.COLOR_RGBA2BGR)

            video_writer.write(mat_img)
            plt.close(fig)
            pbar.update(1)

    video_writer.release()
    print(f"对比视频生成完成: {output_file}")
    return output_file

def save_all_predictions(true_keypoints, pred_keypoints, output_file="predictions.csv", keypoint_scale=1000.0):
    """保存所有预测结果与真实值到CSV文件"""
    import pandas as pd
    import numpy as np

    n_samples = min(len(true_keypoints), len(pred_keypoints))

    columns = []
    for i in range(15):
        columns.extend([f"true_kp{i}_x", f"true_kp{i}_y", f"pred_kp{i}_x", f"pred_kp{i}_y"])

    data = []
    for i in range(n_samples):
        row = []
        true_kp = true_keypoints[i].reshape(15, 2) * keypoint_scale
        pred_kp = pred_keypoints[i].reshape(15, 2) * keypoint_scale

        for j in range(15):
            row.extend([true_kp[j, 0], true_kp[j, 1], pred_kp[j, 0], pred_kp[j, 1]])

        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_file, index=True, index_label="sample_id")

    print(f"已保存所有预测结果到: {output_file}")
    return output_file


def calculate_keypoint_errors(true_keypoints, pred_keypoints, keypoint_scale=1000.0):
    """计算每个关键点的误差统计信息 - 修复版本"""
    import pandas as pd
    import numpy as np

    n_samples = min(len(true_keypoints), len(pred_keypoints))

    # 修复：使用reshape确保正确的形状转换
    true_kp = np.array(true_keypoints[:n_samples]).reshape(n_samples, 15, 2) * keypoint_scale
    pred_kp = np.array(pred_keypoints[:n_samples]).reshape(n_samples, 15, 2) * keypoint_scale

    distances = np.sqrt(np.sum((true_kp - pred_kp) ** 2, axis=2))

    keypoint_stats = []
    for i in range(15):
        kp_distances = distances[:, i]
        stats = {
            'keypoint_id': i,
            'keypoint_name': KEYPOINT_NAMES.get(i, f"关键点 {i}"),
            'body_part': next((part for part, ids in KEYPOINT_GROUPS.items() if i in ids), "未知"),
            'mean_error': np.mean(kp_distances),
            'median_error': np.median(kp_distances),
            'std_error': np.std(kp_distances),
            'min_error': np.min(kp_distances),
            'max_error': np.max(kp_distances)
        }
        keypoint_stats.append(stats)

    df = pd.DataFrame(keypoint_stats)
    return df

def plot_training_history(history, output_dir="vis_results"):
    """绘制训练历史曲线图"""
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(20, 12))

    # 损失曲线
    plt.subplot(2, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], label='Train Total Loss', linewidth=2.5, marker='o', markersize=3)
    plt.plot(epochs, history['val_loss'], label='Val Total Loss', linewidth=2.5, marker='s', markersize=3)
    plt.title('Total Loss', fontsize=15, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 分解损失
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['train_position_loss'], label='Position Loss', linewidth=2, marker='o', markersize=2)
    plt.plot(epochs, history['train_bone_loss'], label='Bone Loss', linewidth=2, marker='s', markersize=2)
    plt.title('Loss Components', fontsize=15, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # MPE曲线
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['train_mpe'], label='Train MPE', linewidth=2.5, marker='o', markersize=3)
    plt.plot(epochs, history['val_mpe'], label='Val MPE', linewidth=2.5, marker='s', markersize=3)
    plt.title('Mean Pose Error', fontsize=15, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MPE', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # PCK曲线
    plt.subplot(2, 3, 4)
    plt.plot(epochs, history['train_pck'], label='Train PCK@0.2', linewidth=2.5, marker='o', markersize=3)
    plt.plot(epochs, history['val_pck'], label='Val PCK@0.2', linewidth=2.5, marker='s', markersize=3)
    plt.title('PCK@0.2 Accuracy', fontsize=15, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('PCK@0.2', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 学习率曲线
    plt.subplot(2, 3, 5)
    plt.plot(epochs, history['lr'], label='Learning Rate', linewidth=2.5, marker='^', markersize=3, color='green')
    plt.title('Learning Rate', fontsize=15, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 损失比例饼图（最后一个epoch）
    plt.subplot(2, 3, 6)
    if len(history['train_position_loss']) > 0:
        last_losses = [
            history['train_position_loss'][-1],
            history['train_bone_loss'][-1]
        ]
        labels = ['Position', 'Bone']
        colors = ['#ff9999', '#66b3ff']
        plt.pie(last_losses, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        plt.title('Final Loss Composition', fontsize=15, fontweight='bold')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 已保存训练历史曲线图到: {output_path}")

    # 保存CSV数据
    history_csv_path = os.path.join(output_dir, 'training_history.csv')
    import pandas as pd
    history_df = pd.DataFrame(history)
    history_df['epoch'] = range(1, len(history_df) + 1)
    history_df.to_csv(history_csv_path, index=False)
    print(f"📊 已保存训练历史数据到: {history_csv_path}")

    return output_path

# ============================== #
# 模型训练函数
# ============================== #
def get_gpu_memory_map():
    """获取所有GPU的显存信息"""
    result = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            result[i] = gpu_memory
    return result

def calculate_optimal_batch_size(gpu_id):
    """根据GPU显存计算最优批量大小"""
    if not torch.cuda.is_available():
        return 32

    gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 ** 3

    if gpu_memory > 40:
        return 64
    elif gpu_memory > 20:
        return 64
    elif gpu_memory > 10:
        return 128
    else:
        return 128

def train_pose_model(train_loader, val_loader, test_loader,
                     batch_size=32, n_epochs=100, patience=5,
                     lr=1e-4, weight_decay=1e-5, keypoint_scale=1000.0,
                     gpu_config='auto', output_dir="vis_results", use_augmentation=False):
    """使用简化损失函数的训练函数"""
    os.makedirs(output_dir, exist_ok=True)

    # GPU配置
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
    print(f"数据增强: {'启用' if use_augmentation else '禁用'}")
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")

    # 批量大小配置
    if torch.cuda.is_available():
        gpu_batch_sizes = [calculate_optimal_batch_size(gpu_id) for gpu_id in gpu_ids]
        physical_batch_size = min(gpu_batch_sizes)
        if len(gpu_ids) == 1 and gpu_ids[0] == 1:
            physical_batch_size = 64
    else:
        physical_batch_size = 64

    gradient_accumulation_steps = max(1, batch_size // (physical_batch_size * len(gpu_ids)))
    effective_batch_size = physical_batch_size * len(gpu_ids) * gradient_accumulation_steps

    print(f"批量配置: 物理批量={physical_batch_size}, GPU数量={len(gpu_ids)}, "
          f"梯度累积={gradient_accumulation_steps}, 有效批量={effective_batch_size}")

    # 初始化模型
    model = HPELINet_Baseline().to(device)

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


    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print("使用DataParallel")

    # 混合精度训练
    scaler = GradScaler()

    # 使用新的简化损失函数
    criterion = SimplePoseLoss(
        position_weight=1.0,
        bone_length_weight=0,  # 增加骨骼长度权重
        loss_type='mse'
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=5e-5,  # 增加权重衰减
        betas=(0.9, 0.999)
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=lr / 1000,
        cooldown=1,
        threshold=1e-4
    )

    # 保存训练历史 - 移除时序相关的项
    history = {
        'train_loss': [], 'val_loss': [],
        'train_position_loss': [], 'train_bone_loss': [],
        'train_mpe': [], 'val_mpe': [],
        'train_pck': [], 'val_pck': [],
        'train_pck50': [], 'val_pck50': [],
        'lr': []
    }

    # 早停参数
    best_val_mpe = float('inf')  # MPE越小越好，所以初始化为无穷大
    patience_counter = 0
    best_model = None
    best_epoch = 0
    best_val_metrics = {'loss': float('inf'), 'mpe': float('inf'), 'pck': 0.0}

    # 重新创建数据加载器
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    val_batch_size = physical_batch_size // 2

    train_loader_optimized = DataLoader(
        train_dataset,
        batch_size=physical_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    val_loader_optimized = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )

    print(f"开始训练，共{n_epochs}个epoch...")

    for epoch in range(n_epochs):
        # ====== 训练阶段 ======
        model.train()

        train_total_loss = 0.0
        train_total_position_loss = 0.0
        train_total_bone_loss = 0.0
        train_total_mpe = 0.0
        train_total_pck = 0.0
        train_total_pck50 = 0.0
        train_samples = 0
        optimizer.zero_grad()

        train_loop = tqdm(train_loader_optimized, desc=f"Epoch {epoch + 1}/{n_epochs} [Train]")
        current_step = 0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loop):
            try:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                # 增强的数据增强
                if use_augmentation and epoch > 0:
                    if torch.rand(1).item() < 0.6:
                        batch_x = time_masking(batch_x.permute(0, 2, 1), mask_ratio=0.3).permute(0, 2, 1)
                    if torch.rand(1).item() < 0.6:
                        batch_x = add_noise(batch_x, noise_level=0.02)
                    if torch.rand(1).item() < 0.5:
                        batch_x = random_scaling(batch_x, scale_range=(0.9, 1.1))

                # 使用混合精度训练
                with autocast():
                    outputs = model(batch_x)
                    loss, loss_dict = criterion(outputs, batch_y)
                    loss = loss / gradient_accumulation_steps

                # 反向传播
                scaler.scale(loss).backward()

                # 计算指标
                with torch.no_grad():
                    mpe = mean_keypoint_error(outputs.detach(), batch_y)
                    pck_results = percentage_correct_keypoints(outputs.detach(), batch_y, thresholds=[0.2, 0.5])
                    pck = pck_results[0.2]
                    pck50 = pck_results[0.5]

                # 累积统计信息
                current_batch_size = batch_y.size(0)
                train_total_loss += (loss.item() * gradient_accumulation_steps) * current_batch_size
                train_total_position_loss += loss_dict['position'] * current_batch_size
                train_total_bone_loss += loss_dict['bone'] * current_batch_size
                train_total_mpe += mpe * current_batch_size
                train_total_pck += pck * current_batch_size
                train_total_pck50 += pck50 * current_batch_size
                train_samples += current_batch_size

                # 更新进度条
                cur_loss = loss.item() * gradient_accumulation_steps
                train_loop.set_postfix(
                    loss=f"{cur_loss:.4f}",
                    mpe=f"{mpe:.4f}",
                    pck20=f"{pck:.4f}",
                    pck50=f"{pck50:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.6f}"
                )

                # 梯度累积
                current_step += 1
                if current_step >= gradient_accumulation_steps:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    current_step = 0

                # 内存清理
                if batch_idx % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "size of tensor a" in str(e) and "must match the size of tensor b" in str(e):
                    print(f"批次 {batch_idx}: 张量大小不匹配，跳过该批次")
                    continue
                else:
                    print(f"训练中遇到错误: {e}")
                    torch.cuda.empty_cache()
                    continue

        # 处理最后一个不完整的累积批次
        if current_step > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()

        # 计算训练指标
        if train_samples > 0:
            train_loss = train_total_loss / train_samples
            train_position_loss = train_total_position_loss / train_samples
            train_bone_loss = train_total_bone_loss / train_samples
            train_mpe = train_total_mpe / train_samples
            train_pck = train_total_pck / train_samples
            train_pck50 = train_total_pck50 / train_samples
        else:
            train_loss = float('inf')
            train_position_loss = train_bone_loss = float('inf')
            train_mpe = float('inf')
            train_pck = train_pck50 = 0.0

        # ====== 验证阶段 ======
        torch.cuda.empty_cache()
        model.eval()

        val_total_loss = 0.0
        val_total_mpe = 0.0
        val_total_pck = 0.0
        val_total_pck50 = 0.0
        val_samples = 0

        val_loop = tqdm(val_loader_optimized, desc=f"Epoch {epoch + 1}/{n_epochs} [Val]")

        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(val_loop):
                try:
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)

                    outputs = model(batch_x)
                    loss, loss_dict = criterion(outputs, batch_y)

                    mpe = mean_keypoint_error(outputs, batch_y)
                    pck_results = percentage_correct_keypoints(outputs, batch_y, thresholds=[0.2, 0.5])
                    pck = pck_results[0.2]
                    pck50 = pck_results[0.5]

                    current_batch_size = batch_y.size(0)
                    val_total_loss += loss.item() * current_batch_size
                    val_total_mpe += mpe * current_batch_size
                    val_total_pck += pck * current_batch_size
                    val_total_pck50 += pck50 * current_batch_size
                    val_samples += current_batch_size

                    val_loop.set_postfix(
                        loss=f"{loss.item():.4f}",
                        mpe=f"{mpe:.4f}",
                        pck20=f"{pck:.4f}",
                        pck50=f"{pck50:.4f}"
                    )

                except RuntimeError as e:
                    if "size of tensor a" in str(e) and "must match the size of tensor b" in str(e):
                        print(f"验证批次 {batch_idx}: 张量大小不匹配，跳过该批次")
                        continue
                    else:
                        print(f"验证出错: {e}")
                        torch.cuda.empty_cache()
                        continue

        # 计算验证指标
        if val_samples > 0:
            val_loss = val_total_loss / val_samples
            val_mpe = val_total_mpe / val_samples
            val_pck = val_total_pck / val_samples
            val_pck50 = val_total_pck50 / val_samples
        else:
            val_loss = float('inf')
            val_mpe = float('inf')
            val_pck = val_pck50 = 0.0

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_position_loss'].append(train_position_loss)
        history['train_bone_loss'].append(train_bone_loss)
        history['train_mpe'].append(train_mpe)
        history['val_mpe'].append(val_mpe)
        history['train_pck'].append(train_pck)
        history['val_pck'].append(val_pck)
        history['train_pck50'].append(train_pck50)
        history['val_pck50'].append(val_pck50)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # 打印详细的损失信息
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"  Train - Total: {train_loss:.4f}, Position: {train_position_loss:.4f}, "
              f"Bone: {train_bone_loss:.4f}")
        print(f"  Train - MPE: {train_mpe:.4f}, PCK@0.2: {train_pck:.4f}, PCK@0.5: {train_pck50:.4f}")
        print(f"  Val - Loss: {val_loss:.4f}, MPE: {val_mpe:.4f}, PCK@0.2: {val_pck:.4f}, PCK@0.5: {val_pck50:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 基于验证损失调整学习率
        scheduler.step(val_mpe)

        # 早停检查
        if val_mpe < best_val_mpe:  # MPE越小越好
            best_val_mpe = val_mpe
            best_val_metrics['pck'] = val_pck
            best_val_metrics['mpe'] = val_mpe
            best_val_metrics['loss'] = val_loss

            if hasattr(model, 'module'):
                best_model = copy.deepcopy(model.module.state_dict())
            else:
                best_model = copy.deepcopy(model.state_dict())

            best_epoch = epoch
            patience_counter = 0

            model_path = os.path.join(output_dir, "best_pose_model.pth")
            torch.save(best_model, model_path)
            print(f"  💾 保存最佳模型 (Epoch {best_epoch + 1}, MPE={val_mpe:.4f}) 到 {model_path}")
        else:
            patience_counter += 1
            print(f"  验证MPE未改善，耐心计数: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"⏹️ 早停在 {epoch + 1} 个epoch后触发。最佳epoch: {best_epoch + 1}")
            break

    # 加载最佳模型进行测试
    if best_model is not None:
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_model)
        else:
            model.load_state_dict(best_model)
        print(f"✅ 加载最佳模型，来自 epoch {best_epoch + 1}")

    print(f"🎯 最佳验证指标 - Loss: {best_val_metrics['loss']:.4f}, "
          f"MPE: {best_val_metrics['mpe']:.4f}, PCK@0.2: {best_val_metrics['pck']:.4f}")

    # 保存训练历史图表
    print("📊 正在生成训练历史曲线图...")
    plot_training_history(history, output_dir)

    # ====== 测试阶段 ======
    test_loader_optimized = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )

    model.eval()

    test_total_loss = 0.0
    test_total_mpe = 0.0
    test_total_pck10 = 0.0
    test_total_pck20 = 0.0
    test_total_pck30 = 0.0
    test_total_pck40 = 0.0
    test_total_pck50 = 0.0
    test_samples = 0
    all_pred_keypoints = []
    all_true_keypoints = []

    test_loop = tqdm(test_loader_optimized, desc="测试中")

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(test_loop):
            try:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                outputs = model(batch_x)
                loss, _ = criterion(outputs, batch_y)

                mpe = mean_keypoint_error(outputs, batch_y)
                pck_results = percentage_correct_keypoints(outputs, batch_y, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5])
                pck10 = pck_results[0.1]
                pck20 = pck_results[0.2]
                pck30 = pck_results[0.3]
                pck40 = pck_results[0.4]
                pck50 = pck_results[0.5]

                current_batch_size = batch_y.size(0)
                test_total_loss += loss.item() * current_batch_size
                test_total_mpe += mpe * current_batch_size
                test_total_pck10 += pck10 * current_batch_size
                test_total_pck20 += pck20 * current_batch_size
                test_total_pck30 += pck30 * current_batch_size
                test_total_pck40 += pck40 * current_batch_size
                test_total_pck50 += pck50 * current_batch_size
                test_samples += current_batch_size

                all_pred_keypoints.append(outputs.cpu().numpy())
                all_true_keypoints.append(batch_y.cpu().numpy())

                test_loop.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mpe=f"{mpe:.4f}",
                    pck20=f"{pck20:.4f}",
                    pck50=f"{pck50:.4f}"
                )

            except RuntimeError as e:
                if "size of tensor a" in str(e) and "must match the size of tensor b" in str(e):
                    print(f"测试批次 {batch_idx}: 张量大小不匹配，跳过该批次")
                    continue
                else:
                    print(f"测试时出错: {e}")
                    torch.cuda.empty_cache()
                    continue

    # 计算测试指标
    if test_samples > 0:
        test_loss = test_total_loss / test_samples
        test_mpe = test_total_mpe / test_samples
        test_pck10 = test_total_pck10 / test_samples
        test_pck20 = test_total_pck20 / test_samples
        test_pck30 = test_total_pck30 / test_samples
        test_pck40 = test_total_pck40 / test_samples
        test_pck50 = test_total_pck50 / test_samples
    else:
        test_loss = float('inf')
        test_mpe = float('inf')
        test_pck10 = test_pck20 = test_pck30 = test_pck40 = test_pck50 = 0.0

    # 显示所有PCK阈值的测试结果
    print(f"🎯 测试结果:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   MPE: {test_mpe:.4f}")
    print(f"   PCK@0.1: {test_pck10:.4f}")
    print(f"   PCK@0.2: {test_pck20:.4f}")
    print(f"   PCK@0.3: {test_pck30:.4f}")
    print(f"   PCK@0.4: {test_pck40:.4f}")
    print(f"   PCK@0.5: {test_pck50:.4f}")

    # 保存预测结果和生成视频
    if all_pred_keypoints and all_true_keypoints:
        all_preds = np.vstack(all_pred_keypoints)
        all_trues = np.vstack(all_true_keypoints)

        # 保存预测结果
        predictions_file = os.path.join(output_dir, "test_predictions.csv")
        save_all_predictions(all_trues, all_preds, predictions_file, keypoint_scale)
        print(f"💾 已保存测试预测结果到: {predictions_file}")

        # 计算关键点误差统计
        error_stats = calculate_keypoint_errors(
            all_trues[:min(1000, len(all_trues))],
            all_preds[:min(1000, len(all_preds))],
            keypoint_scale=keypoint_scale
        )
        error_stats_file = os.path.join(output_dir, "keypoint_error_stats.csv")
        error_stats.to_csv(error_stats_file)
        print(f"📊 已保存关键点误差统计到: {error_stats_file}")

        # 保存详细的测试结果到CSV
        test_results_file = os.path.join(output_dir, "test_results_summary.csv")
        test_results_data = {
            'Metric': ['Loss', 'MPE', 'PCK@0.1', 'PCK@0.2', 'PCK@0.3', 'PCK@0.4', 'PCK@0.5'],
            'Value': [test_loss, test_mpe, test_pck10, test_pck20, test_pck30, test_pck40, test_pck50]
        }
        import pandas as pd
        test_results_df = pd.DataFrame(test_results_data)
        test_results_df.to_csv(test_results_file, index=False)
        print(f"📊 已保存测试结果汇总到: {test_results_file}")

        # 生成视频
        try:
            videos_dir = os.path.join(output_dir, "videos")
            os.makedirs(videos_dir, exist_ok=True)

            frames_to_animate = min(720, len(all_preds))
            print(f"正在为前{frames_to_animate}帧生成视频...")

            # 1. 创建真实姿态视频
            print("正在生成真实姿态视频...")
            true_subset = all_trues[:frames_to_animate].copy()
            true_animation = create_pose_animation_opencv(
                true_subset,
                output_file=os.path.join(videos_dir, "true_poses.mp4"),
                keypoint_scale=keypoint_scale,
                show_labels=True
            )
            print(f"已生成真实姿态视频: {true_animation}")

            # 2. 创建预测姿态视频
            print("正在生成预测姿态视频...")
            pred_subset = all_preds[:frames_to_animate].copy()
            pred_animation = create_pose_animation_opencv(
                pred_subset,
                output_file=os.path.join(videos_dir, "predicted_poses.mp4"),
                keypoint_scale=keypoint_scale,
                show_labels=True
            )
            print(f"已生成预测姿态视频: {pred_animation}")

            # 3. 创建对比视频
            print("正在生成对比视频...")
            comparison_video = create_side_by_side_video_opencv(
                true_subset,
                pred_subset,
                output_file=os.path.join(videos_dir, "comparison_poses.mp4"),
                keypoint_scale=keypoint_scale,
                fps=30
            )
            print(f"已生成对比视频: {comparison_video}")

            print(f"已完成所有视频的生成，保存在 {videos_dir} 目录下")

        except Exception as e:
            print(f"生成视频时出错: {e}")
            import traceback
            traceback.print_exc()

    return model, history, test_loss, test_pck20, test_mpe, {
        'pck10': test_pck10,
        'pck20': test_pck20,
        'pck30': test_pck30,
        'pck40': test_pck40,
        'pck50': test_pck50
    }


def main():
    # 定义统一的输出目录
    output_dir = "hpeli"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"所有结果将保存到目录: {output_dir}")

    # 添加命令行参数
    parser = argparse.ArgumentParser(description='训练CSI姿态估计模型')
    parser.add_argument('--gpu', type=str, default='1',
                        help='GPU配置: auto(自动选择), 1,2(使用GPU0和GPU2), 1(仅使用GPU1)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='逻辑批量大小（梯度累积后的有效批量）')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--output_dir', type=str, default=output_dir,
                        help=f'输出目录 (默认: {output_dir})')
    parser.add_argument('--use_augmentation', action='store_true', default=False,
                        help='是否使用数据增强 (默认: 不使用)')
    args = parser.parse_args()

    # 如果用户通过命令行指定了输出目录，则使用用户指定的
    if args.output_dir != output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"使用用户指定的输出目录: {output_dir}")

    # 设置随机种子确保可重复性
    set_seed(42)

    # 显示系统信息
    print(f"系统内存使用情况: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}, 显存: {props.total_memory / 1024 ** 3:.1f}GB")

    # 关键点归一化因子
    keypoint_scale = 1000.0

    # 预处理数据目录
    preprocessed_dir = "preprocessed_csi_data"  # CSI预处理数据
    pam_label_dir = "keypoints_pam_data"  # PAM格式标签目录

    # 检查预处理数据是否存在
    if not os.path.exists(preprocessed_dir) or not os.path.exists(os.path.join(preprocessed_dir, "csi_windows.npy")):
        print(f"错误: 未找到预处理数据 {preprocessed_dir}")
        print("请先运行 preprocess_csi_data.py 脚本生成预处理数据")
        return

    # 使用预处理数据创建数据集
    try:
        print(f"正在加载预处理数据...")
        dataset = PreprocessedCSIKeypointsDataset(
            data_dir="preprocessed_csi_data",
            keypoint_scale=1000.0,
            enable_temporal_clean=True
        )

        # 创建数据加载器
        train_loader, val_loader, test_loader = create_preprocessed_train_val_test_loaders(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=0
        )

        # 查看一个批次的数据形状
        for csi_batch, keypoints_batch in train_loader:
            print(f"CSI数据形状: {csi_batch.shape}")
            print(f"关键点数据形状: {keypoints_batch.shape}")
            if torch.isnan(csi_batch).any() or torch.isinf(csi_batch).any():
                print("警告: CSI数据包含NaN或Inf")
            if torch.isnan(keypoints_batch).any() or torch.isinf(keypoints_batch).any():
                print("警告: 关键点数据包含NaN或Inf")
            break

    except Exception as e:
        print(f"创建数据加载器时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # 训练参数
    n_epochs = args.epochs
    # 改成3-5比较好，在epoch15之后过拟合精度下降
    patience = 5
    lr = args.lr
    weight_decay = 1e-5

    # 训练模型
    print(f"开始训练模型...")
    print(f"GPU配置: {args.gpu}")
    print(f"批量大小: {args.batch_size}")
    print(f"训练轮数: {n_epochs}")
    print(f"学习率: {lr}")
    print(f"输出目录: {output_dir}")

    try:
        # 修复：正确接收所有返回值
        model, history, test_loss, test_pck, test_mpe, pck_details = train_pose_model(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            batch_size=args.batch_size,
            n_epochs=n_epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            keypoint_scale=keypoint_scale,
            gpu_config=args.gpu,
            output_dir=output_dir,
            use_augmentation=args.use_augmentation
        )

        print(f"训练完成，测试损失: {test_loss:.4f}, 测试PCK@0.2: {test_pck:.4f}")
        print(f"详细PCK结果: {pck_details}")
        print(f"所有结果已保存到: {output_dir}")

    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

# 命令行入口点
if __name__ == "__main__":
    # 设置多进程启动方法
    if torch.cuda.is_available():
        try:
            torch.multiprocessing.set_start_method('spawn', force=True)
            gc.collect()
            torch.cuda.empty_cache()
        except RuntimeError:
            pass

    main()