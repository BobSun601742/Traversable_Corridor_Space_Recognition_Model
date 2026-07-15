import numpy as np
import torch


def logits_to_costmap(logits):
    """
    logits -> cost map (0~1)
    0 = safe, 1 = dangerous
    """
    probs = torch.sigmoid(logits)
    cost = probs.squeeze().cpu().numpy()
    return cost


def compute_best_direction(cost_map):
    h, w = cost_map.shape

    # 1. 可通行区域（低 cost）
    binary = (cost_map < 0.5).astype(np.uint8)

    ys, xs = np.where(binary == 1)

    if len(xs) < 50:
        return 0  # fallback

    # 2. 上下分区
    top_mask = ys < h * 0.4
    bottom_mask = ys > h * 0.6

    if np.sum(top_mask) < 10 or np.sum(bottom_mask) < 10:
        # fallback：用整体方向
        cx, cy = w // 2, h // 2
        cx_mean, cy_mean = xs.mean(), ys.mean()
        dx = cx_mean - cx
        dy = cy_mean - cy
        return np.degrees(np.arctan2(dx, -dy))

    # 3. 计算重心
    top_center = (xs[top_mask].mean(), ys[top_mask].mean())
    bottom_center = (xs[bottom_mask].mean(), ys[bottom_mask].mean())

    dx = top_center[0] - bottom_center[0]
    dy = top_center[1] - bottom_center[1]

    angle = np.degrees(np.arctan2(dx, -dy))

    return angle