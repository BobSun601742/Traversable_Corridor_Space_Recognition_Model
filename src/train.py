import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import CorridorSegDataset
from src.model import TinyUNet
from src.utils import load_config, dice_loss, safety_loss, smoothness_loss, island_loss, iou_score

def train():
    cfg = load_config("config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    image_size = (
        cfg["train"]["image_width"],
        cfg["train"]["image_height"],
    )

    train_dataset = CorridorSegDataset(
        cfg["data"]["train_images"],
        cfg["data"]["train_masks"],
        image_size=image_size,
    )

    val_dataset = CorridorSegDataset(
        cfg["data"]["val_images"],
        cfg["data"]["val_masks"],
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    model = TinyUNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
    ).to(device)

    bce_loss = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
    )

    checkpoint_dir = cfg["save"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_iou = 0.0

    for epoch in range(cfg["train"]["epoches"]):
        model.train()

        train_loss = 0.0
        train_iou = 0.0

        loop = tqdm(train_loader, desc=f"Epoch[{epoch+1}/{cfg['train']['epoches']}]")

        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            logits = model(images)

            loss = (
                bce_loss(logits, masks)
                + dice_loss(logits, masks)
                + safety_loss(logits, masks)
                + 0.05 * smoothness_loss(logits)
                + 0.75 * island_loss(logits, masks)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_iou = iou_score(logits.detach(), masks.detach())

            train_loss += loss.item()
            train_iou += batch_iou 

            loop.set_postfix(loss=loss.item(), iou=batch_iou)

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)

                loss = bce_loss(logits, masks) + dice_loss(logits, masks)
                batch_iou = iou_score(logits, masks)

                val_loss += loss.item()
                val_iou += batch_iou

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, train_iou={train_iou:.4f}, "
            f"val_loss={val_loss:.4f}, val_iou={val_iou:.4f}"
        )

        latest_path = os.path.join(checkpoint_dir, "latest.pth")
        torch.save(model.state_dict(), latest_path)

        if val_iou > best_iou:
            best_iou = val_iou
            best_path = os.path.join(checkpoint_dir, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}, IoU={best_iou:.4f}")


if __name__ == "__main__":
    train()