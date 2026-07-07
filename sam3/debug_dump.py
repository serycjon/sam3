# Copyright (c) 2026 Jonas Serych
"""
Writer for streaming-tracker debug dumps.

Saves, per tracked frame, a thumbnail, the output mask, and (optionally) the
multimask/token0 candidates, plus the tracker's debug log — everything the
`scripts/memory_viz` server needs to visualize a run without the model.
See `scripts/memory_viz/PLAN.md` for the dump format.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np


class DebugDumpWriter:
    """
    Incrementally write a memory-visualization dump during tracking.

    Example:
        tracker = SAM3StreamingTracker(debug=True)
        dump = DebugDumpWriter("runs/seq01_dump")
        mask = tracker.init(frame0, init_mask)
        dump.add_frame(0, frame0, mask)
        for i, frame in enumerate(frames, start=1):
            mask, cands = tracker.track(frame, return_all_masks=True)
            dump.add_frame(i, frame, mask, candidates=cands)
        dump.finalize(tracker)
    """

    def __init__(
        self,
        out_dir: Union[str, Path],
        thumb_width: int = 384,
        jpeg_quality: int = 85,
    ) -> None:
        """
        Args:
            out_dir: Dump directory (created if missing).
            thumb_width: Width of the saved thumbnails in pixels (aspect kept);
                masks and candidate masks are stored at the same resolution.
            jpeg_quality: JPEG quality for the thumbnails.
        """
        self.out_dir = Path(out_dir)
        for sub in ("thumbs", "masks", "multimask"):
            (self.out_dir / sub).mkdir(parents=True, exist_ok=True)
        self.thumb_width = thumb_width
        self.jpeg_quality = jpeg_quality
        # Per-frame scalar metadata for the candidates (IoUs, token0 iou/stability),
        # accumulated here and written as a single small JSON in finalize(); the masks
        # themselves go to disk as binary PNGs in add_frame().
        self._mm_meta: Dict[int, Dict[str, Any]] = {}

    def _thumb_hw(self, frame_shape: Any) -> "tuple[int, int]":
        h, w = frame_shape[:2]
        tw = min(self.thumb_width, w)
        th = max(1, round(h * tw / w))
        return th, tw

    def add_frame(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        mask: Optional[np.ndarray] = None,
        candidates: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save one frame's thumbnail, output mask, and optional candidates.

        Args:
            frame_idx: Absolute frame index (same indexing as the debug log).
            frame_bgr: The frame as passed to the tracker (HxWx3 uint8 BGR).
            mask: The output mask returned by init()/track() (HxW bool).
            candidates: The dict returned by ``track(return_all_masks=True)``
                (multimask masks/ious and optional token0 entries).
        """
        th, tw = self._thumb_hw(frame_bgr.shape)
        thumb = cv2.resize(frame_bgr, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.imwrite(
            str(self.out_dir / "thumbs" / f"{frame_idx:06d}.jpg"),
            thumb,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )

        if mask is not None:
            self._save_mask_png(mask, tw, th, "masks", f"{frame_idx:06d}.png")

        # Store the candidate masks as binary PNGs (what the viz thresholds them to
        # anyway); keep only the scalar IoU/stability metadata for the sidecar JSON.
        if candidates is not None and candidates.get("masks") is not None:
            masks = np.asarray(candidates["masks"])  # (M, H, W) bool; M may be 0
            if masks.size > 0:
                for k in range(masks.shape[0]):
                    self._save_mask_png(
                        masks[k], tw, th, "multimask", f"{frame_idx:06d}_{k}.png"
                    )
                meta: Dict[str, Any] = {
                    "ious": [float(x) for x in np.asarray(candidates["ious"])],
                    "n_candidates": int(masks.shape[0]),
                }
                if "token0_mask" in candidates:
                    self._save_mask_png(
                        candidates["token0_mask"],
                        tw,
                        th,
                        "multimask",
                        f"{frame_idx:06d}_token0.png",
                    )
                    meta["token0_iou"] = float(candidates["token0_iou"])
                    meta["token0_stability"] = float(candidates["token0_stability"])
                self._mm_meta[frame_idx] = meta

    def _save_mask_png(
        self, mask: np.ndarray, tw: int, th: int, sub: str, name: str
    ) -> None:
        """Threshold a mask to 0/255, resize to thumb resolution, and write a PNG."""
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
        mask_thumb = cv2.resize(mask_u8, (tw, th), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(self.out_dir / sub / name), mask_thumb)

    def finalize(self, tracker: Any) -> None:
        """Write ``debug.json`` and ``multimask.json`` from the tracker + candidates."""
        if not getattr(tracker, "debug", False):
            raise ValueError(
                "DebugDumpWriter.finalize needs a SAM3StreamingTracker(debug=True); "
                "the tracker has no debug log to save"
            )
        payload = {"config": tracker.debug_config, "log": tracker.debug_log}
        with open(self.out_dir / "debug.json", "w") as f:
            json.dump(payload, f)
        with open(self.out_dir / "multimask.json", "w") as f:
            json.dump(self._mm_meta, f)
