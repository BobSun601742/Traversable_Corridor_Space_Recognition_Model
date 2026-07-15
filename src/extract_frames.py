import os
import cv2
import random


def ensure_dirs():
    os.makedirs("data/train_images", exist_ok=True)
    os.makedirs("data/val_images", exist_ok=True)
    os.makedirs("data/test_images", exist_ok=True)


def extract_frames(video_path, counters, probs, every_n=60):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n[video] {video_path}")
    print(f"[info] total_frames={total_frames}, fps={fps:.2f}, every_n={every_n}")

    frame_id = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % every_n == 0:
            r = random.random()

            if r < probs["train"]:
                split = "train"
            elif r < probs["train"] + probs["val"]:
                split = "val"
            else:
                split = "test"

            idx = counters[split]
            filename = f"{split}_{idx:05d}.jpg"
            out_path = os.path.join(f"data/{split}_images", filename)

            ok = cv2.imwrite(out_path, frame)
            if not ok:
                print(f"[warn] failed to save: {out_path}")
            else:
                print(f"{video_path} -> {split}: {filename}")
                counters[split] += 1
                saved_count += 1

        frame_id += 1

    cap.release()
    print(f"[done] {video_path}, saved {saved_count} frames")


if __name__ == "__main__":

    random.seed(42)

    ensure_dirs()

    videos = [
        "data/videos/video1.avi",
        "data/videos/video2.avi",
        "data/videos/video3.avi",
        "data/videos/video4.avi",
    ]

    counters = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    probs = {
        "train": 0.75,
        "val": 0.15,
        "test": 0.10,
    }

    for v in videos:
        extract_frames(
            video_path=v,
            counters=counters,
            probs=probs,
            every_n=10
        )

    print("\n[DONE] dataset created!")
    print("[count] train:", counters["train"])
    print("[count] val:", counters["val"])
    print("[count] test:", counters["test"])