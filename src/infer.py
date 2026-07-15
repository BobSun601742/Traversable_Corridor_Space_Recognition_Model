import os
import sys
import glob

import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TinyUNet
from src.utils import load_config
from src.planner import logits_to_costmap, compute_best_direction
from src.visualize import draw_overlay, draw_direction


# =========================
# GT mask loader
# =========================
def load_gt_mask(image_path, mask_dir, size_hw):
    base = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(mask_dir, base + ".png")

    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(mask_path)

    gt = cv2.resize(gt, (size_hw[1], size_hw[0]))  # (W, H)
    gt = (gt > 127).astype(np.uint8)
    return gt


def infer_one(image_path, model, device, cfg, output_dir):
    image = Image.open(image_path).convert("RGB")
    original = np.array(image)

    image_height = cfg["train"]["image_height"]
    image_width = cfg["train"]["image_width"]

    resized = TF.resize(image, (image_height, image_width))
    tensor = TF.to_tensor(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)[0, 0].cpu()

    # =========================
    # costmap
    # =========================
    cost_map = logits_to_costmap(logits.unsqueeze(0))

    # =========================
    # best direction
    # =========================
    angle = compute_best_direction(cost_map)

    # =========================
    # visualization
    # =========================
    overlay = draw_overlay(original, cost_map)
    final_vis = draw_direction(overlay, angle)

    # =========================
    # save
    # =========================
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    mask_path = os.path.join(output_dir, base_name + "_cost.png")
    vis_path = os.path.join(output_dir, base_name + "_nav.png")

    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(mask_path, (cost_map * 255).astype(np.uint8))
    cv2.imwrite(vis_path, cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR))

    # =========================
    # GT evaluation
    # =========================
    gt_mask = load_gt_mask(
        image_path,
        "data/test_masks",
        (image_height, image_width)
    )

    pred_mask = (cost_map > 0.5).astype(np.uint8)
    pred_mask = pred_mask.squeeze()

    correct = (pred_mask == gt_mask).sum()
    total = gt_mask.size
    acc = correct / total

    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = inter / (union + 1e-6)

    print(f"[OK] {base_name} → angle: {angle:.2f}° | acc: {acc:.4f} | IoU: {iou:.4f}")

    return acc, iou


def main():
    cfg = load_config("config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =========================
    # model
    # =========================
    model = TinyUNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
    ).to(device)

    checkpoint_path = "runs/checkpoints/best.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # =========================
    # data
    # =========================
    image_list = sorted(glob.glob("data/test_images/*.jpg"))
    output_dir = cfg["save"]["output_dir"]

    # =========================
    # metrics accumulator
    # =========================
    all_acc = []
    all_iou = []

    for image_path in image_list:
        acc, iou = infer_one(image_path, model, device, cfg, output_dir)
        all_acc.append(acc)
        all_iou.append(iou)

    # =========================
    # summary
    # =========================
    print("\n========================")
    print("      FINAL RESULT")
    print("========================")
    print(f"Mean Accuracy: {np.mean(all_acc):.4f}")
    print(f"Mean IoU     : {np.mean(all_iou):.4f}")


if __name__ == "__main__":
    main()