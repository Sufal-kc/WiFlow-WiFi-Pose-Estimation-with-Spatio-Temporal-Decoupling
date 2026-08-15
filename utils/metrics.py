import torch
import numpy as np


def calculate_pck(pred, target, thresholds=[0.2], use_torso_norm=True):
    """计算PCK (针对 baseline 两端点)

    pred/target: [B,4] 即 [x1,y1,x2,y2]
    threshold: 相对于目标线长度的比值
    """
    batch_size = pred.shape[0]

    if len(pred.shape) == 2 and pred.shape[1] == 30:
        pred = pred.reshape(batch_size, 15, 2)
        target = target.reshape(batch_size, 15, 2)
        # 将关键点转换为两端点（同样如 dataset 中的做法）
        pred = _keypoints_batch_to_endpoints(pred)
        target = _keypoints_batch_to_endpoints(target)

    # 现在 pred/target 为 [B,4]
    pred_pts = pred.view(batch_size, 2, 2)
    target_pts = target.view(batch_size, 2, 2)

    # 线长度作为归一化因子
    line_lengths = torch.sqrt(torch.sum((target_pts[:, 0, :] - target_pts[:, 1, :]) ** 2, dim=1))
    line_lengths = torch.clamp(line_lengths, min=1e-2)

    dists = torch.sqrt(torch.sum((pred_pts - target_pts) ** 2, dim=2))  # [B,2]
    normalized = dists / line_lengths.unsqueeze(1)

    pck_results = {}
    for threshold in thresholds:
        correct = (normalized <= threshold).float()
        pck_results[threshold] = float(correct.mean().item())

    return pck_results


def calculate_mpjpe(pred, target):
    """计算平均端点误差（Mean Per Joint Position Error）针对 baseline 两端点"""
    batch_size = pred.shape[0]

    if len(pred.shape) == 2 and pred.shape[1] == 30:
        pred = pred.reshape(batch_size, 15, 2)
        target = target.reshape(batch_size, 15, 2)
        pred = _keypoints_batch_to_endpoints(pred)
        target = _keypoints_batch_to_endpoints(target)

    pred_pts = pred.view(batch_size, 2, 2)
    target_pts = target.view(batch_size, 2, 2)

    dists = torch.sqrt(torch.sum((pred_pts - target_pts) ** 2, dim=2))
    mean_distance = torch.mean(dists)
    return float(mean_distance.item())


def _keypoints_batch_to_endpoints(kp_batch: torch.Tensor) -> torch.Tensor:
    """将 (B,15,2) 的关键点批次转换为 (B,4) 的端点: [x_min,y_at_xmin,x_max,y_at_xmax]"""
    out = []
    kp_np = kp_batch.detach().cpu().numpy()
    for kp in kp_np:
        mask = ~((kp[:, 0] == 0) & (kp[:, 1] == 0))
        pts = kp[mask]
        if pts.shape[0] < 2:
            out.append([0, 0, 0, 0])
        else:
            a, b = np.polyfit(pts[:, 0], pts[:, 1], 1)
            x_min = float(pts[:, 0].min())
            x_max = float(pts[:, 0].max())
            out.append([x_min, a * x_min + b, x_max, a * x_max + b])

    return torch.tensor(out, dtype=kp_batch.dtype, device=kp_batch.device)