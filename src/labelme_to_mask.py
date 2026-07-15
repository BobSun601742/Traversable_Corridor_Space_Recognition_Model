import os
import json
import cv2
import numpy as np
from PIL import Image


def create_mask_for_image(image_path, json_path, output_mask_path):
    image = Image.open(image_path)
    width, height = image.size

    mask = np.zeros((height, width), dtype=np.uint8)

    if json_path is not None and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for shape in data.get("shapes", []):
            label = shape.get("label", "")

            if label != "walkable":
                continue

            points = np.array(shape["points"], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

    cv2.imwrite(output_mask_path, mask)


def convert_folder(image_dir, output_mask_dir):
    os.makedirs(output_mask_dir, exist_ok=True)

    image_names = [
        name for name in os.listdir(image_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    count = 0

    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        base_name = os.path.splitext(image_name)[0]

        json_path = os.path.join(image_dir, base_name + ".json")
        output_mask_path = os.path.join(output_mask_dir, base_name + ".png")

        create_mask_for_image(image_path, json_path, output_mask_path)
        count += 1

    print(f"Converted {count} images from {image_dir} to {output_mask_dir}")


if __name__ == "__main__":
    convert_folder(
        image_dir="data/train_images",
        output_mask_dir="data/train_masks",
    )

    convert_folder(
        image_dir="data/test_images",
        output_mask_dir="data/test_masks",
    )

    convert_folder(
        image_dir="data/val_images",
        output_mask_dir="data/val_masks",
    )