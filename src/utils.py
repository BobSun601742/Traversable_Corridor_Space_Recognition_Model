import yaml
import torch
import torch.nn.functional as F

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)

    probs = probs.view(-1)
    targets = targets.view(-1)

    intersection = (probs * targets).sum()
    dice = (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

    return 1.0 - dice

def safety_loss(logits, targets):
    probs = torch.sigmoid(logits)

    fp = probs * (1 - targets)
    fn = (1 - probs) * targets

    fp_loss = fp.pow(2).mean()
    fn_loss = fn.mean()

    return 5.0 * fp_loss + 1.0 * fn_loss

def smoothness_loss(logits):
    prob = torch.sigmoid(logits)

    dx = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1])
    dy = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :])

    return dx.mean() + dy.mean()

def island_loss(logits, masks, kernel_size=5):
    prob = torch.sigmoid(logits)

    false_positive = prob * (1.0 - masks)

    neighbor_mean = F.avg_pool2d(
        prob,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    )

    isolated_fp = false_positive * (1.0 - neighbor_mean)

    return isolated_fp.mean()

def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    return ((intersection + smooth) / (union + smooth)).item()
