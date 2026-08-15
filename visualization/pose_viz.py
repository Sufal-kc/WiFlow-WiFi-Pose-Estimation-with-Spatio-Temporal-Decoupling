import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from tqdm import tqdm

def create_side_by_side_video_opencv(true_keypoints, pred_keypoints, output_file="comparison.mp4",
                                     keypoint_scale=1.0, fps=30):
    """创建对比视频"""
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from tqdm import tqdm

    # 现在 true_keypoints / pred_keypoints 为 [T,4] 格式: [x1,y1,x2,y2]
    frames = min(len(true_keypoints), len(pred_keypoints))
    true_lines = np.array(true_keypoints[:frames]).reshape(frames, 4)
    pred_lines = np.array(pred_keypoints[:frames]).reshape(frames, 4)

    if keypoint_scale != 1.0:
        true_lines *= keypoint_scale
        pred_lines *= keypoint_scale

    # 计算全局范围
    all_x = np.concatenate([true_lines[:, [0, 2]].flatten(), pred_lines[:, [0, 2]].flatten()])
    all_y = np.concatenate([true_lines[:, [1, 3]].flatten(), pred_lines[:, [1, 3]].flatten()])

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

            # 真实基线
            x1, y1, x2, y2 = true_lines[frame_idx]
            ax1.plot([x1, x2], [y1, y2], color='blue', linewidth=4)
            ax1.scatter([x1, x2], [y1, y2], c=['green', 'red'], s=80)

            ax1.set_xlim(x_min - x_margin, x_max + x_margin)
            ax1.set_ylim(y_max + y_margin, y_min - y_margin)
            ax1.set_title(f"True Baseline - Frame {frame_idx + 1}", fontsize=14)
            ax1.set_aspect('equal')
            ax1.axis('off')

            # 预测基线
            px1, py1, px2, py2 = pred_lines[frame_idx]
            ax2.plot([px1, px2], [py1, py2], color='orange', linewidth=4)
            ax2.scatter([px1, px2], [py1, py2], c=['green', 'red'], s=80)

            ax2.set_xlim(x_min - x_margin, x_max + x_margin)
            ax2.set_ylim(y_max + y_margin, y_min - y_margin)
            ax2.set_title(f"Predicted Baseline - Frame {frame_idx + 1}", fontsize=14)
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

    columns = ['true_x1', 'true_y1', 'true_x2', 'true_y2', 'pred_x1', 'pred_y1', 'pred_x2', 'pred_y2']
    data = []

    for i in range(n_samples):
        true_line = np.array(true_keypoints[i]).reshape(4,) * keypoint_scale
        pred_line = np.array(pred_keypoints[i]).reshape(4,) * keypoint_scale
        row = [float(true_line[0]), float(true_line[1]), float(true_line[2]), float(true_line[3]),
               float(pred_line[0]), float(pred_line[1]), float(pred_line[2]), float(pred_line[3])]
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_file, index=True, index_label="sample_id")

    print(f"已保存所有预测结果到: {output_file}")
    return output_file


def calculate_keypoint_errors(true_keypoints, pred_keypoints, keypoint_scale=1000.0):
    """计算每个关键点的误差统计信息 - 修复版本"""
    import pandas as pd
    import numpy as np

    # 现在输入为 baseline endpoints: [N,4]
    n_samples = min(len(true_keypoints), len(pred_keypoints))
    true_lines = np.array(true_keypoints[:n_samples]).reshape(n_samples, 4) * keypoint_scale
    pred_lines = np.array(pred_keypoints[:n_samples]).reshape(n_samples, 4) * keypoint_scale

    # 计算每个端点的距离 (两端)
    true_pts = true_lines.reshape(n_samples, 2, 2)
    pred_pts = pred_lines.reshape(n_samples, 2, 2)
    distances = np.sqrt(np.sum((true_pts - pred_pts) ** 2, axis=2))  # [N,2]

    stats = []
    for endpoint_idx in range(2):
        d = distances[:, endpoint_idx]
        stats.append({
            'endpoint': endpoint_idx,
            'mean_error': float(np.mean(d)),
            'median_error': float(np.median(d)),
            'std_error': float(np.std(d)),
            'min_error': float(np.min(d)),
            'max_error': float(np.max(d))
        })

    df = pd.DataFrame(stats)
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