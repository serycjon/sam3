# SAM 3 — Streaming API fork

A fork of Meta's SAM 3 (Segment Anything with Concepts) that adds a small,
high-level streaming single-object tracking API on top of the original model,
plus the ability to correct the segmentation mid-stream.

Upstream SAM 3 tracks video through an offline-style predictor that expects the
whole clip up front. This fork adds `sam3.SAM3StreamingTracker`, an
`init → track → track → …` interface for live or unbounded streams where frames
arrive one at a time, memory stays bounded on long videos, and a corrected mask
can be fed back mid-stream to become authoritative conditioning going forward.

The streaming layer lives in [`sam3/streaming_tracker.py`](sam3/streaming_tracker.py)
and is exported as `from sam3 import SAM3StreamingTracker`. Everything under
`sam3/model/` is the upstream model; see the
[upstream repo](https://github.com/facebookresearch/sam3) for the paper, model
card, benchmarks, and the SAM 3.1 checkpoints.

Streaming additions by Jonas Serych.

## Installation

Following upstream (Python ≥ 3.12, recent PyTorch + CUDA):

```bash
conda create -n sam3 python=3.12 && conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/serycjon/sam3.git
cd sam3
pip install -e .
```

Request access to the checkpoints on the SAM 3 Hugging Face
[repo](https://huggingface.co/facebook/sam3) and authenticate (`hf auth login`)
before first use — the model is downloaded on the first `SAM3StreamingTracker()`
construction.

## Demo

[`examples/demo.py`](examples/demo.py) runs the streaming tracker on a
forward-backward cycling video and writes overlay frames to `streaming_demo_out/`:

```bash
python examples/demo.py
```

Two flags inject periodic corrections as sanity checks (corrected frames are
tinted green in the overlay): `--correct-self` feeds the tracker's own mask back
unchanged (expected to be a no-op), and `--correct-shift` feeds a deliberately
shifted mask of growing offset (the tracker should follow the corruption). The
demo also prints an FPS / GPU / CPU memory report at the end.

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

When the returned mask is wrong, hand-annotate a corrected mask for the current
frame (the one most recently returned by `track()`) and feed it back. It becomes
authoritative conditioning and subsequent frames use it as memory:

```python
mask = tracker.track(frame)
if user_sees_a_problem:
    corrected = annotate(frame)        # HxW bool, your own UI / tool
    tracker.correct(frame, corrected)
mask = tracker.track(next_frame)
```

The tracker is single-object (`obj_id = 1`). Frames are OpenCV `HxWx3 uint8 BGR`
arrays; masks are `HxW` booleans.

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
| `keep_first_cond_frame` | `True` | Always keep the first-frame annotation among the conditioning frames attended to, so tracking can't drift away from the original object. (Upstream default is `False`.) |
| `accumulate_corrections` | `False` | If `False`, conditioning frames that can never be re-selected for attention are evicted to free GPU memory. If `True`, every correction is kept forever. |
| `clear_recent_memory_on_correct` | `False` | If `True`, drop recent non-conditioning memory around a correction so tracking leans on the corrected frame. Turn on when an error persisted for many frames before being corrected. |

## How it works

The streaming tracker is a thin orchestration layer over the upstream tracker
predictor. Two things make a frame-at-a-time stream work:

- `_direct` variants of the predictor methods (`add_new_mask_direct`,
  `propagate_in_video_single`) take the raw image for the current frame and
  compute its features on the fly, instead of indexing into a clip of frames
  pre-loaded into the inference state as the offline path expects. `init` and
  `correct` are both the same prompt path (`add_new_mask_direct` +
  `propagate_in_video_preflight`), applied at frame 0 and at the current frame
  respectively.

- Bounded storage. Recent non-conditioning memory is trimmed every frame to what
  the temporal memory-selection logic would still pick. Conditioning frames (the
  initial mask and corrections) are never attended to beyond the closest
  `max_cond_frames_in_attn`, and since the stream only moves forward, any
  conditioning frame that drops out of that window can never re-enter it — so
  with `accumulate_corrections=False` it is evicted to free GPU memory (keeping
  the first frame when `keep_first_cond_frame=True`).

## Relationship to upstream

This fork tracks `facebookresearch/sam3`. The upstream model code under
`sam3/model/` is unmodified except for the streaming helpers
(`add_new_mask_direct`, `propagate_in_video_single`) and minor build fixes:

```bash
git remote add fb_upstream https://github.com/facebookresearch/sam3.git
git fetch fb_upstream
```

## Note on authorship

The streaming layer was written by hand, but from the mid-stream correction
commit (`b272205d`) onward this repo is partially vibecoded — parts were produced
with AI coding assistance. Review accordingly.

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
