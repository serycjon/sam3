import os
import glob
import time
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

def count_frames(video_path):
    """Number of frames in the input (an .mp4 file or a directory of .jpg)."""
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n
    return len(glob.glob(os.path.join(str(video_path), "*.jpg")))

def gpu_mem_mb():
    """Current and peak CUDA memory (allocated, reserved) in MiB, or zeros on CPU."""
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    mib = 1024 * 1024
    return (
        torch.cuda.memory_allocated() / mib,
        torch.cuda.memory_reserved() / mib,
        torch.cuda.max_memory_allocated() / mib,
    )

def cpu_rss_mb():
    """Resident set size of this process in MiB (reads /proc on Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # value is in kB
    except OSError:
        pass
    # Fallback: ru_maxrss is peak (kB on Linux), not current, but better than nothing.
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def main():
    data_dir = Path(__file__).parent.parent / 'assets' / 'videos'
    video_path = data_dir / "0001"
    annot_path = data_dir / "0001_init_mask.png"
    init_mask = cv2.imread(annot_path, cv2.IMREAD_GRAYSCALE) > 0

    out_dir = Path('streaming_demo_out')
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = SAM3StreamingTracker()

    n_frames = count_frames(video_path)

    # Resource instrumentation: per-frame tracker latency (for FPS) plus periodic
    # GPU/CPU memory samples, so we can confirm the streaming tracker stays flat.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    track_times = []  # seconds spent in the tracker per non-init frame
    mem_samples = []  # (frame_idx, gpu_alloc_mb, cpu_rss_mb) sampled over the run
    sample_every = 100

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Burn-in: ignore the first forward pass for the flat/growing verdict. During
    # that pass the recent-memory bank fills up and the GPU/CPU footprint legitimately
    # ramps; only after it do we expect the streaming footprint to plateau.
    burn_in_frames = max(0, n_frames)

    try:
        for frame_idx, frame in tqdm.tqdm(enumerate(forward_backward(load_frames(video_path)))):
            sync()
            t0 = time.perf_counter()
            if frame_idx == 0:
                mask = tracker.init(frame, init_mask)
                initialized = True
            else:
                mask = tracker.track(frame)
            sync()
            # Skip frame 0 (init) — it is a one-off outlier, not steady-state tracking.
            if frame_idx > 0:
                track_times.append(time.perf_counter() - t0)

            vis = frame.copy()
            vis[mask, 2] = 255

            out_path = out_dir / f'{frame_idx:05d}.jpg'
            cv2.imwrite(str(out_path), vis)

            # Sample memory periodically (and always on the very first frame).
            if frame_idx == 0 or frame_idx % sample_every == 0:
                gpu_alloc, _, _ = gpu_mem_mb()
                mem_samples.append((frame_idx, gpu_alloc, cpu_rss_mb()))

            if frame_idx > 50000:
                break
    except KeyboardInterrupt:
        print("\ninterrupted (Ctrl-C) — reporting on frames processed so far")
    finally:
        _print_report(track_times, mem_samples, burn_in_frames)

def _print_report(track_times, mem_samples, burn_in_frames=0):
    """Print a short FPS / GPU / CPU memory report and whether memory grew."""
    print("\n===== streaming resource report =====")

    if track_times:
        arr = np.array(track_times)
        fps = 1.0 / arr.mean()
        print(
            f"frames timed : {len(track_times)} "
            f"(init frame excluded)"
        )
        print(
            f"FPS          : {fps:6.2f} avg  "
            f"(per-frame {arr.mean() * 1e3:.1f} ms avg, "
            f"{np.median(arr) * 1e3:.1f} ms median, "
            f"{arr.max() * 1e3:.1f} ms max)"
        )
    else:
        print("FPS          : n/a (no frames tracked)")

    _, _, gpu_peak = gpu_mem_mb()
    if mem_samples:
        first_f, first_gpu, first_cpu = mem_samples[0]
        last_f, last_gpu, last_cpu = mem_samples[-1]
        print(
            f"GPU alloc    : {first_gpu:8.1f} MiB @f{first_f} -> "
            f"{last_gpu:8.1f} MiB @f{last_f}  (peak {gpu_peak:.1f} MiB)"
        )
        print(
            f"CPU RSS      : {first_cpu:8.1f} MiB @f{first_f} -> "
            f"{last_cpu:8.1f} MiB @f{last_f}"
        )

        # Verdict over the post-burn-in window only: the first forward pass fills
        # the memory bank, so measuring growth from frame 0 would always look like
        # growth. Compare from the first sample taken after burn_in_frames instead.
        post = [s for s in mem_samples if s[0] >= burn_in_frames]
        if len(post) >= 2:
            b_f, b_gpu, b_cpu = post[0]
            e_f, e_gpu, e_cpu = post[-1]
            gpu_growth = e_gpu - b_gpu
            cpu_growth = e_cpu - b_cpu

            # Heuristic verdict: flat if growth is small relative to the footprint.
            def _verdict(growth, base):
                return "FLAT" if abs(growth) <= max(50.0, 0.05 * base) else "GROWING"

            print(
                f"post burn-in : frames {b_f}..{e_f} (burn-in {burn_in_frames})  "
                f"GPU {gpu_growth:+.1f} MiB, CPU {cpu_growth:+.1f} MiB"
            )
            print(
                f"verdict      : GPU {_verdict(gpu_growth, b_gpu)}, "
                f"CPU {_verdict(cpu_growth, b_cpu)}"
            )
        else:
            print(
                f"verdict      : n/a (need >=2 samples after burn-in of "
                f"{burn_in_frames} frames; got {len(post)})"
            )
    else:
        print("memory       : n/a (no samples)")
    print("=====================================")

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
