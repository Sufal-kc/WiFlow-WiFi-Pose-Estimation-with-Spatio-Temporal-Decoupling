import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PoseLoss(nn.Module):
    """Baseline-line 损失函数：仅对两个端点进行回归损失计算"""

    def __init__(self, loss_type: str = 'smooth_l1'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 4] or [B, 30]（若为30则会被reshape并转换为端点）
            target: [B, 4] or [B, 30]
        Returns:
            total_loss, loss_dict
        """
        batch_size = pred.shape[0]

        # 支持旧格式 (30) -> 提取端点 (x1,y1,x2,y2) 假设顺序为 (kp0_x,kp0_y,...)
        if len(pred.shape) == 2 and pred.shape[1] == 30:
            pred = pred.reshape(batch_size, 15, 2)
            # 将关键点转换为 baseline：简单地用最小和最大 x 投影拟合直线
            pred_np = pred.detach().cpu().numpy()
            new_preds = []
            for i in range(batch_size):
                kp = pred_np[i]
                mask = ~((kp[:, 0] == 0) & (kp[:, 1] == 0))
                pts = kp[mask]
                if pts.shape[0] < 2:
                    new_preds.append([0, 0, 0, 0])
                else:
                    a, b = np.polyfit(pts[:, 0], pts[:, 1], 1)
                    x_min = float(pts[:, 0].min())
                    x_max = float(pts[:, 0].max())
                    new_preds.append([x_min, a * x_min + b, x_max, a * x_max + b])
            pred = torch.tensor(new_preds, device=pred.device, dtype=pred.dtype)

        if len(target.shape) == 2 and target.shape[1] == 30:
            target = target.reshape(batch_size, 15, 2)
            target_np = target.detach().cpu().numpy()
            new_tgts = []
            for i in range(batch_size):
                kp = target_np[i]
                mask = ~((kp[:, 0] == 0) & (kp[:, 1] == 0))
                pts = kp[mask]
                if pts.shape[0] < 2:
                    new_tgts.append([0, 0, 0, 0])
                else:
                    a, b = np.polyfit(pts[:, 0], pts[:, 1], 1)
                    x_min = float(pts[:, 0].min())
                    x_max = float(pts[:, 0].max())
                    new_tgts.append([x_min, a * x_min + b, x_max, a * x_max + b])
            target = torch.tensor(new_tgts, device=target.device, dtype=target.dtype)

        # 现在 pred 和 target 应为 [B,4]
        if self.loss_type == 'mse':
            position_loss = F.mse_loss(pred, target)
        elif self.loss_type == 'l1':
            position_loss = F.l1_loss(pred, target)
        elif self.loss_type == 'smooth_l1':
            position_loss = F.smooth_l1_loss(pred, target, beta=0.1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        loss_dict = {'position': position_loss.item()}
        total_loss = position_loss

        return total_loss, loss_dict