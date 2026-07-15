import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class CorridorSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256,144)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        self.image_names = sorted([
            name for name in os.listdir(image_dir)
            if name.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        if len(self.image_names) == 0:
            raise RuntimeError(f"No images found in {image_dir}")
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)

        base_name = os.path.splitext(image_name)[0]
        possible_mask_names = [
            base_name + ".png",
            base_name + ".jpg",
            base_name + ".jpeg",
        ]

        mask_path = None
        for mask_name in possible_mask_names:
            candidate = os.path.join(self.mask_dir, mask_name)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            print("image_name:", image_name)
            print("base_name:", base_name)
            print("mask_dir:", self.mask_dir)
            print("mask_dir abs:", os.path.abspath(self.mask_dir))
            print("mask files first 10:", os.listdir(self.mask_dir)[:10])
            raise FileNotFoundError(f"No mask found for image: {image_name}")
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, self.image_size[::-1])
        mask = TF.resize(mask, self.image_size[::-1], interpolation=Image.NEAREST)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return image, mask