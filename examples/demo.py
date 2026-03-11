import os
import glob
from pathlib import Path

import tqdm
import einops
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sam3 import SAM3StreamingTracker

def load_frames(video_path):
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        video_frames_for_vis = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
            # video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
        try:
            # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
            video_frames_for_vis.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
        except ValueError:
            # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
            print(
                f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
                f"falling back to lexicographic sort."
            )
            video_frames_for_vis.sort()
        for path in video_frames_for_vis:
            img = cv2.imread(str(path))
            yield img

def forward_backward(gen):
    """
    A generator that cycles forward-backward-forward-backward over another generator.
    """
    buffer = []
    # 1st phase: collecting and yielding forward
    for item in gen:
        buffer.append(item)
        yield item
    
    if not buffer:
        return  # nothing to yield

    # 2nd phase: forward-backward cycle indefinitely
    while True:
        # Backward, skip first and last to avoid repeats
        for item in reversed(buffer[1:-1]):
            yield item
        # Forward again
        for item in buffer:
            yield item

def repeat_first(gen):
    item = next(gen)

    while True:
        yield item

def main():
    data_dir = Path(__file__).parent.parent / 'assets' / 'videos'
    video_path = data_dir / "0001"
    annot_path = data_dir / "0001_init_mask.png"
    init_mask = cv2.imread(annot_path, cv2.IMREAD_GRAYSCALE) > 0

    out_dir = Path('streaming_demo_out')
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = SAM3StreamingTracker()

    for frame_idx, frame in tqdm.tqdm(enumerate(forward_backward(load_frames(video_path)))):
        if frame_idx == 0:
            mask = tracker.init(frame, init_mask)
            initialized = True
        else:
            mask = tracker.track(frame)

        vis = frame.copy()
        vis[mask, 2] = 255

        out_path = out_dir / f'{frame_idx:05d}.jpg'
        cv2.imwrite(str(out_path), vis)

        if frame_idx > 50000:
            break

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    main()
