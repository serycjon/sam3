import os
import glob
import time
import argparse
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

# Correction sanity-check parameters.
# We aim for a correction roughly every TARGET frames, but pick the actual
# interval at runtime so it does not cleanly divide the input length (see
# pick_correction_interval).
TARGET_INTERVAL = 100
# How much the shift magnitude grows on every correction (in pixels).
SHIFT_STEP = 3
# Seed for the per-correction random shift direction (fixed for repeatability).
SHIFT_SEED = 0
# Cap the shift magnitude at this fraction of the smaller image dimension, so the
# corrupted mask usually stays at least partly inside the frame.
SHIFT_CAP_FRAC = 0.5

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

def pick_correction_interval(length, target=TARGET_INTERVAL):
    """
    Pick a correction interval close to ``target`` that does not evenly divide
    ``length`` (so a correction never lands on the same phase every cycle, even
    for adversarial lengths like a multiple of the target).

    Searches outward from ``target`` (target, target-1, target+1, ...) and
    returns the first candidate that leaves a non-zero remainder.
    """
    if length <= 0:
        return target
    for delta in range(0, target):
        for cand in (target - delta, target + delta):
            if cand > 1 and length % cand != 0:
                return cand
    return target  # pathological fallback (e.g. length == 1)

# Distinct BGR colors for rendering candidate masks in the --return-all-masks
# overlay (the model predicts up to 3-4 candidates per frame).
CANDIDATE_COLORS = [
    (0, 0, 255),    # red
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
]

def overlay_candidates(vis, candidates, alpha=0.5):
    """
    Alpha-blend each candidate mask onto ``vis`` in a distinct color and draw a
    legend with each candidate's predicted IoU. The best-IoU candidate (the one
    the tracker actually outputs) is marked with a '*'.

    Args:
        vis: HxWx3 uint8 BGR image, modified in place and returned.
        candidates: dict from ``tracker.track(..., return_all_masks=True)`` with
            "masks" (M,H,W) bool and "ious" (M,) float arrays.
        alpha: blend strength for the colored overlay.
    """
    masks = candidates["masks"]
    ious = candidates["ious"]
    if masks.shape[0] == 0:
        return vis  # no candidates for this frame (e.g. object absent)
    best = int(np.argmax(ious))
    for i in range(masks.shape[0]):
        color = np.array(CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)], dtype=np.float32)
        m = masks[i]
        vis[m] = ((1.0 - alpha) * vis[m] + alpha * color).astype(np.uint8)
    for i in range(masks.shape[0]):
        color = CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)]
        marker = "*" if i == best else " "
        text = f"{marker}cand {i}: IoU {ious[i]:.3f}"
        cv2.putText(
            vis, text, (10, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 1, cv2.LINE_AA,
        )
    return vis

def shift_mask(mask, dy, dx):
    """
    Translate a boolean mask by (dy, dx) pixels, filling exposed border with False.

    Args:
        mask: HxW boolean numpy array.
        dy: vertical shift in pixels (positive moves the mask down).
        dx: horizontal shift in pixels (positive moves the mask right).

    Returns:
        Shifted HxW boolean numpy array (no wraparound).
    """
    out = np.zeros_like(mask)
    h, w = mask.shape
    # Source/destination slice bounds for each axis given the (possibly negative) shift.
    src_y = slice(max(0, -dy), h - max(0, dy))
    dst_y = slice(max(0, dy), h - max(0, -dy))
    src_x = slice(max(0, -dx), w - max(0, dx))
    dst_x = slice(max(0, dx), w - max(0, -dx))
    out[dst_y, dst_x] = mask[src_y, src_x]
    return out

def main(correct_self=False, correct_shift=False, return_all_masks=False):
    data_dir = Path(__file__).parent.parent / 'assets' / 'videos'
    video_path = data_dir / "0001"
    annot_path = data_dir / "0001_init_mask.png"
    init_mask = cv2.imread(annot_path, cv2.IMREAD_GRAYSCALE) > 0

    out_dir = Path('streaming_demo_out')
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = SAM3StreamingTracker()

    # Pick a correction interval near TARGET_INTERVAL that does not cleanly
    # divide the input length, so corrections do not lock onto one phase.
    n_frames = count_frames(video_path)
    correction_interval = pick_correction_interval(n_frames)
    if correct_self or correct_shift:
        print(
            f"{n_frames} input frames; correcting every {correction_interval} frames"
        )

    # Number of corrections applied so far; the shift magnitude grows as
    # N * SHIFT_STEP pixels (capped) for the --correct-shift sanity check.
    n_corrections = 0
    # Seeded RNG so the random shift directions are repeatable across runs.
    shift_rng = np.random.default_rng(SHIFT_SEED)

    # Resource instrumentation: per-frame tracker latency (for FPS) plus periodic
    # GPU/CPU memory samples, so we can confirm the streaming tracker stays flat.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    track_times = []  # seconds spent in the tracker per non-init frame
    mem_samples = []  # (frame_idx, gpu_alloc_mb, cpu_rss_mb) sampled over the run
    sample_every = max(1, correction_interval)

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Burn-in: ignore the warm-up phase for the flat/growing verdict. Two things
    # ramp legitimately before the footprint plateaus:
    #   1. the first forward pass fills the recent-memory bank (~n_frames);
    #   2. corrections accumulate conditioning frames until eviction kicks in. The
    #      tracker keeps at most max_cond_frames_in_attn corrections, and one lands
    #      every correction_interval frames, so eviction only starts steady-state
    #      after ~(N + 1) corrections.
    burn_in_frames = max(0, n_frames)
    if correct_self or correct_shift:
        n_kept = tracker.predictor.max_cond_frames_in_attn
        if n_kept == -1:
            # Unbounded attention: corrections are never evicted, so the footprint
            # has no plateau to find. Use the whole run as warm-up (verdict -> n/a).
            burn_in_frames = max(burn_in_frames, 51000)
            print(
                "warning: max_cond_frames_in_attn == -1 (corrections never evicted); "
                "memory is expected to grow, flat/growing verdict not meaningful"
            )
        else:
            corrections_to_saturate = n_kept + 1
            burn_in_frames = max(
                burn_in_frames, corrections_to_saturate * correction_interval
            )

    try:
        for frame_idx, frame in tqdm.tqdm(enumerate(forward_backward(load_frames(video_path)))):
            sync()
            t0 = time.perf_counter()
            candidates = None
            if frame_idx == 0:
                mask = tracker.init(frame, init_mask)
                initialized = True
            elif return_all_masks:
                mask, candidates = tracker.track(frame, return_all_masks=True)
            else:
                mask = tracker.track(frame)
            sync()
            # Skip frame 0 (init) — it is a one-off outlier, not steady-state tracking.
            if frame_idx > 0:
                track_times.append(time.perf_counter() - t0)

            # Periodically feed a correction back into the tracker as a sanity check.
            is_correction_frame = (
                (correct_self or correct_shift)
                and frame_idx > 0
                and frame_idx % correction_interval == 0
            )
            if is_correction_frame:
                if correct_shift:
                    # Take the current frame's tracker output and deliberately corrupt
                    # it by shifting it in a random (seeded) direction. The magnitude
                    # grows N * SHIFT_STEP pixels each correction but is capped near half
                    # the image so the mask usually stays partly in frame. The shifted
                    # mask is fed as the "correction" and shown as the demo output.
                    n_corrections += 1
                    h, w = mask.shape
                    max_shift = int(min(h, w) * SHIFT_CAP_FRAC)
                    magnitude = min(n_corrections * SHIFT_STEP, max_shift)
                    angle = shift_rng.uniform(0.0, 2.0 * np.pi)
                    dy = int(round(magnitude * np.sin(angle)))
                    dx = int(round(magnitude * np.cos(angle)))
                    mask = shift_mask(mask, dy, dx)
                # --correct-self feeds the unmodified SAM mask straight back.
                tracker.correct(mask)

            vis = frame.copy()
            if candidates is not None:
                # Render every candidate mask in its own color (best one marked).
                vis = overlay_candidates(vis, candidates)
            else:
                # Tint corrected frames green and normal tracked frames red (BGR), so
                # it is obvious at a glance which frames a correction was applied to.
                tint_channel = 1 if is_correction_frame else 2
                vis[mask, tint_channel] = 255

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
    parser = argparse.ArgumentParser(description="SAM3 streaming demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--correct-self",
        action="store_true",
        help=(
            f"Every ~{TARGET_INTERVAL} frames, feed the current SAM mask back as "
            "a correction (sanity check: should be a no-op)."
        ),
    )
    group.add_argument(
        "--correct-shift",
        action="store_true",
        help=(
            f"Every ~{TARGET_INTERVAL} frames, feed a deliberately shifted mask as "
            f"the correction and demo output: a seeded random direction with magnitude "
            f"growing {SHIFT_STEP}px each time, capped near half the image (sanity "
            "check: tracker should follow the corruption)."
        ),
    )
    parser.add_argument(
        "--return-all-masks",
        action="store_true",
        help=(
            "Render all of SAM's per-frame candidate masks (before best-mask "
            "selection), each in a distinct color, with their predicted IoUs in a "
            "legend (the best-IoU candidate, the tracker's actual output, is marked "
            "'*')."
        ),
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    main(
        correct_self=args.correct_self,
        correct_shift=args.correct_shift,
        return_all_masks=args.return_all_masks,
    )
