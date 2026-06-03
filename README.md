# SAM 3 — Streaming API fork

A patched fork of Meta's **SAM 3: Segment Anything with Concepts** that adds a
small, high-level **streaming single-object tracking API** on top of the original
model, plus the ability to **correct the segmentation mid-stream**.

- Upstream project: <https://github.com/facebookresearch/sam3>
- Upstream paper / model / benchmarks: see the original repo above (this README
  intentionally omits the upstream results tables, model card and author list —
  refer to upstream for those).

Streaming additions by **Jonas Serych**. Everything under `sam3/model/` is the
upstream model; the streaming layer is a thin wrapper around it.

---

## What this fork adds

Upstream SAM 3 tracks video through an offline-style predictor that expects the
whole clip up front (or a session against a fixed resource) and propagates across
a known frame range. This fork adds **`sam3.SAM3StreamingTracker`** — a minimal
`init → track → track → …` interface designed for:

- **Live / unbounded streams** — frames arrive one at a time; nothing is buffered
  ahead of time.
- **Long videos** — memory is trimmed continuously so cost stays bounded instead
  of growing with video length.
- **Single-object mask tracking** — initialize from a binary mask, get a binary
  mask per frame.
- **Mid-stream correction** — when tracking drifts, hand a corrected mask back to
  the tracker and it becomes authoritative conditioning going forward, without
  restarting the session.

The streaming layer lives in [`sam3/streaming_tracker.py`](sam3/streaming_tracker.py)
and is exported as `from sam3 import SAM3StreamingTracker`. It is supported by a
few additions to the upstream tracker predictor
([`sam3/model/sam3_tracking_predictor.py`](sam3/model/sam3_tracking_predictor.py)),
namely `add_new_mask_direct` and `propagate_in_video_single` (frame-at-a-time
variants of the upstream batched methods).

---

## Installation

Same as upstream (Python ≥ 3.12, PyTorch ≥ 2.7, CUDA ≥ 12.6):

```bash
conda create -n sam3 python=3.12 && conda activate sam3
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

⚠️ Request access to the checkpoints on the SAM 3 Hugging Face
[repo](https://huggingface.co/facebook/sam3) and authenticate
(`hf auth login`) before first use — the model is downloaded from there on the
first `SAM3StreamingTracker()` construction.

---

## Usage

```python
import cv2
from sam3 import SAM3StreamingTracker

tracker = SAM3StreamingTracker()

# Initialize with the first frame (HxWx3 uint8 BGR) and a binary mask (HxW bool).
first_frame = cv2.imread("frame_0000.jpg")
init_mask = cv2.imread("init_mask.png", cv2.IMREAD_GRAYSCALE) > 0
tracker.init(first_frame, init_mask)

# Track subsequent frames; each call returns a HxW boolean mask.
for frame in stream:                 # frames as they arrive
    mask = tracker.track(frame)
    # ... use mask ...
```

### Correcting drift mid-stream

When the returned mask is wrong, hand-annotate a corrected mask for the **current**
frame (the one most recently returned by `track()`) and feed it back:

```python
mask = tracker.track(frame)
if user_sees_a_problem:
    corrected = annotate(frame)      # HxW bool, your own UI / tool
    tracker.correct(frame, corrected)  # becomes authoritative conditioning
# continue tracking; subsequent frames use the correction as memory
mask = tracker.track(next_frame)
```

### Constructor options

```python
SAM3StreamingTracker(
    keep_first_cond_frame=True,          # pin the initial mask in attention
    accumulate_corrections=False,        # evict corrections that can't be re-used
    clear_recent_memory_on_correct=False # keep recent history on correction
)
```

| Flag | Default | Effect |
|------|---------|--------|
| `keep_first_cond_frame` | `True` | Always keep the first-frame annotation among the conditioning frames attended to, so tracking can't drift away from the original object identity. (Upstream model default is `False`.) |
| `accumulate_corrections` | `False` | If `False`, conditioning frames that can never be re-selected for attention again are evicted to free GPU memory (see below). If `True`, every correction is kept forever. |
| `clear_recent_memory_on_correct` | `False` | If `True`, drop recent non-conditioning memory around a correction so tracking leans on the corrected frame. Default keeps recent temporal history and lets the (conditioning) corrected frame dominate. Turn on when an error persisted for many frames before being corrected. |

### Demo

[`examples/demo.py`](examples/demo.py) runs the streaming tracker on a
forward-backward cycling video and writes overlay frames to `streaming_demo_out/`:

```bash
python examples/demo.py
```

---

## How it works

The streaming tracker is a thin orchestration layer over the upstream tracker
predictor. The full implementation is ~160 lines in
[`sam3/streaming_tracker.py`](sam3/streaming_tracker.py).

### Initialization (`init`)
1. `init_state(...)` with a small **dummy frame count** (the upstream video model
   expects a frame count up front; the streaming tracker doesn't know the true
   length, so it passes a placeholder).
2. `add_new_mask_direct(frame_idx=0, mask=...)` — adds the first-frame mask as a
   **conditioning frame**. The `_direct` variant takes the raw frame and computes
   image features on the fly (the offline path expects frames to be pre-loaded
   into the inference state).
3. `propagate_in_video_preflight(...)` — consolidates the prompt and runs the
   **memory encoder** on the first frame. (Upstream defers the memory encoder out
   of `add_new_mask*` so non-overlap constraints can be applied across objects
   first; preflight is where it actually runs.)

### Per frame (`track`)
- `propagate_in_video_single(frame, frame_idx)` — a frame-at-a-time version of
  upstream's `propagate_in_video` loop: it computes features for the new frame,
  conditions on the memory bank, predicts the mask, and encodes the result into
  memory.
- After each frame, `_trim_memory(...)` deletes non-conditioning memory the
  tracker's own memory-selection logic would no longer pick, keeping memory
  bounded on long streams. Conditioning frames are never trimmed.

### Correction (`correct`)
A correction is just the `init` prompt path applied at the **current** frame
instead of frame 0:
1. `add_new_mask_direct(frame_idx=self.frame_idx, mask=corrected)` adds the mask
   as a conditioning frame.
2. `propagate_in_video_preflight(...)` consolidates it and runs the memory encoder
   **only on that new frame** (cheap). It also replaces the frame's previously
   tracked output, so the corrected mask supersedes it.
3. Stale conditioning frames are optionally evicted (see below).

### Memory & conditioning frames — the bounded-storage story
SAM 3's memory bank distinguishes:
- **Non-conditioning memory** — recent tracked frames; bounded by `num_maskmem`
  (7 in this build) via the temporal memory-selection logic, and trimmed each
  frame by `_trim_memory`.
- **Conditioning frames** — user-prompted frames (the initial mask and every
  correction). At attention time the model only attends to the
  `max_cond_frames_in_attn` (4 in this build) temporally-closest conditioning
  frames, but it does **not** free the rest — they accumulate on the GPU.

Because the stream only ever moves forward, a conditioning frame that drops out of
the "closest 4" window can never re-enter it. With `accumulate_corrections=False`
(default), the streaming tracker therefore **evicts** any conditioning frame that
can no longer be selected, freeing its GPU memory. With `keep_first_cond_frame=True`
the protected set is `{first frame} ∪ {the most-recent corrections}`; otherwise it
is simply the most-recent conditioning frames.

### Notes / technicalities
- **Single object.** The tracker hardcodes `obj_id = 1`.
- **Frame format.** Frames are OpenCV `HxWx3 uint8 BGR` arrays; masks are `HxW`
  booleans.
- **`correct()` targets the current frame only.** The streaming tracker keeps no
  past frames, so corrections apply to the frame just returned by `track()`; the
  caller passes that frame's image alongside the corrected mask.
- **Mid-stream `propagate_in_video_preflight` is safe.** It is idempotent: it
  consolidates and memory-encodes only the newly added correction frame, and its
  internal bookkeeping assertions stay satisfied because the correction is added
  to the input-frame set and the consolidated set together.
- See [`TERNARY_MASK_ANALYSIS.md`](TERNARY_MASK_ANALYSIS.md) (untracked scratch
  notes) for an analysis of whether a three-valued object/background/ignore
  correction mask is feasible.

---

## Relationship to upstream

This fork tracks `facebookresearch/sam3`. The upstream model code under
`sam3/model/` is unmodified except for the streaming helpers
(`add_new_mask_direct`, `propagate_in_video_single`) and minor build fixes. To pull
upstream updates:

```bash
git remote add fb_upstream https://github.com/facebookresearch/sam3.git
git fetch fb_upstream
```

---

## Note on authorship

The streaming layer was written by hand, but from the **mid-stream correction
commit** (`b272205d`) onward this repo is **partially vibecoded** — parts were
produced with AI coding assistance. Review accordingly.

## License

Licensed under the SAM License — see [LICENSE](LICENSE). Upstream model, weights,
and the SA-Co dataset are governed by their respective terms on the
[original repository](https://github.com/facebookresearch/sam3).

## Citing SAM 3

```bibtex
@misc{carion2025sam3segmentconcepts,
      title={SAM 3: Segment Anything with Concepts},
      author={Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman Rädle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
      year={2025},
      eprint={2511.16719},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16719},
}
```
