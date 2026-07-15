import cv2
import numpy as np


def draw_overlay(image, cost_map):
    """
    把 cost map 叠加到原图
    """

    heat = (cost_map * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    heat = cv2.resize(heat, (image.shape[1], image.shape[0]))

    overlay = cv2.addWeighted(image, 0.6, heat, 0.4, 0)
    return overlay


def draw_direction(image, angle):
    h, w = image.shape[:2]

    cx, cy = w // 2, h - 10
    length = 80

    dx = int(length * np.sin(np.radians(angle)))
    dy = int(-length * np.cos(np.radians(angle)))

    cv2.arrowedLine(
        image,
        (cx, cy),
        (cx + dx, cy + dy),
        (0, 255, 0),
        3,
        tipLength=0.3
    )

    return image