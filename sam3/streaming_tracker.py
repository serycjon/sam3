# Copyright (c) 2026 Jonas Serych
"""
Streaming SAM3 tracker for long videos and video streams.

This module provides a high-level interface for single-object tracking in video streams,
optimized for memory efficiency and suitable for long-running tracking tasks.
"""

import json
from typing import Any, Dict, List, Optional

import einops
import numpy as np
import torch


class SAM3StreamingTracker:
    """
    High-level streaming interface for SAM3 single-object tracking.
    
    This class provides a simple init/track interface for video tracking that:
    - Operates on OpenCV frames (numpy arrays in BGR format)
    - Manages memory efficiently for long videos
    - Tracks a single object through a video stream
    
    Example usage:
        tracker = SAM3StreamingTracker()
        
        # Initialize with first frame and mask
        mask = tracker.init(first_frame, initial_mask)
        
        # Track in subsequent frames
        for frame in video_frames:
            mask = tracker.track(frame)
    """

    def __init__(
        self,
        keep_first_cond_frame: bool = True,
        accumulate_corrections: bool = False,
        clear_recent_memory_on_correct: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Initialize the streaming tracker with SAM3 model.

        Loads the SAM3 video model and sets up the tracking predictor.

        Args:
            keep_first_cond_frame: Pin the initial first-frame annotation so it is
                always among the conditioning frames attended to, even after several
                corrections (guards against drifting away from the original object).
                The bare SAM3 model defaults this to False; we default it to True for
                streaming since we never go back to re-prompt the first frame.
            accumulate_corrections: If True, keep every correction's conditioning
                frame forever. If False (default), evict correction frames that can
                never be re-selected for attention (the stream only moves forward),
                freeing their GPU memory.
            clear_recent_memory_on_correct: If True, drop the recent non-conditioning
                memory just before a correction so subsequent tracking relies on the
                corrected frame instead of the (possibly polluted) recent history.
                Default False: keep recent temporal history and let the corrected
                conditioning frame dominate. Turn on when an error persisted for many
                frames before being corrected.
            debug: If True, append one compact JSON-serializable record per
                ``init()``/``track()``/``correct()`` call to ``self.debug_log``,
                describing the memory-attention roster actually used for the frame,
                the frame's quality scores, and all memory-bank mutations (trims,
                evictions). Pure Python scalars only (no tensors); see
                ``save_debug_log`` for persisting it.
        """
        from sam3.model_builder import build_sam3_video_model

        sam3_model = build_sam3_video_model()
        predictor = sam3_model.tracker
        predictor.backbone = sam3_model.detector.backbone
        predictor.keep_first_cond_frame = keep_first_cond_frame

        self.predictor = predictor
        self.obj_id = 1
        self.accumulate_corrections = accumulate_corrections
        self.clear_recent_memory_on_correct = clear_recent_memory_on_correct
        self.debug = debug
        # Only in debug mode, have the mask decoder stash the single-mask token's
        # output (token 0), which the multimask tracking path otherwise discards;
        # exposed via track(return_all_masks=True). Kept off by default so the decoder
        # never retains an extra per-frame tensor tuple on the normal tracking path.
        predictor.sam_mask_decoder.expose_token0_output = debug
        predictor.mem_debug_enabled = debug
        self.debug_log: List[Dict[str, Any]] = []
        self.debug_config: Dict[str, Any] = {
            "num_maskmem": predictor.num_maskmem,
            "max_obj_ptrs_in_encoder": predictor.max_obj_ptrs_in_encoder,
            "max_cond_frames_in_attn": predictor.max_cond_frames_in_attn,
            "mf_threshold": predictor.mf_threshold,
            "memory_temporal_stride_for_eval": predictor.memory_temporal_stride_for_eval,
            "use_memory_selection": predictor.use_memory_selection,
            "keep_first_cond_frame": keep_first_cond_frame,
            "accumulate_corrections": accumulate_corrections,
            "clear_recent_memory_on_correct": clear_recent_memory_on_correct,
        }
        self.inference_state = None
        # Latest frame seen by init()/track(), kept so correct() can reuse it
        # without the caller passing the frame in again (one frame, no growth).
        self._last_frame = None

    def init(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Initialize tracking with the first frame and initial mask.
        
        Args:
            frame: OpenCV frame (HxWx3 numpy array in uint8 BGR format)
            mask: Binary segmentation mask (HxW boolean numpy array)
                  True/1 indicates the object, False/0 indicates background
        
        Returns:
            The input mask (passed through for convenience)
        """
        # This is a workaround for the video model expecting a frame count.
        # Must be >= max_obj_ptrs_in_encoder (16) so the object-pointer memory
        # budget and the sine temporal-position-encoding normalizer match batch
        # behavior, and must exceed any real stream length because num_frames also
        # upper-bounds pointer lookups when memory selection is off.
        DUMMY_N_FRAMES = 1_000_000_000
        
        self.inference_state = self.predictor.init_state(
            video_height=frame.shape[0],
            video_width=frame.shape[1],
            num_frames=DUMMY_N_FRAMES,
        )
        # Mark the state as streaming: it never populates `frames_already_tracked`
        # or `output_dict_per_obj` (see `propagate_in_video_single`), which only
        # mask prompts tolerate. `add_new_points_or_box` refuses to run on it.
        self.inference_state["streaming"] = True
        self.frame_idx = 0
        self._last_frame = frame.copy()
        # init() starts a fresh sequence, so drop any debug records accumulated from
        # a previous sequence tracked with this same tracker instance (the log only
        # ever appends otherwise, which would concatenate sequences in one dump).
        self.debug_log = []

        # Add the initial mask
        self.predictor.add_new_mask_direct(
            inference_state=self.inference_state,
            frame_idx=self.frame_idx,
            obj_id=self.obj_id,
            frame=self._last_frame,
            mask=torch.from_numpy(mask.copy()),
        )
        
        # Prepare for tracking
        self.predictor.propagate_in_video_preflight(self.inference_state)

        self._record_debug("init")

        return mask

    def track(self, frame: np.ndarray, return_all_masks: bool = False):
        """
        Track the object in the next frame.

        Args:
            frame: OpenCV frame (HxWx3 numpy array in uint8 BGR format)
            return_all_masks: If True, also return the model's candidate masks for this
                frame (before best-mask selection) and their predicted IoUs.

        Returns:
            By default, the binary segmentation mask (HxW boolean numpy array; True/1
            indicates the object, False/0 background).

            If ``return_all_masks=True``, a tuple ``(mask, candidates)`` where
            ``candidates`` is a dict with:
              - ``"masks"``: boolean array of shape (M, H, W), the M candidate masks
                for the tracked object thresholded at logit 0 (M is 3 when multimask
                output is active, else 1);
              - ``"logits"``: float32 array of shape (M, H, W), the raw mask logits;
              - ``"ious"``: float32 array of shape (M,), the predicted IoU of each
                candidate (the default output mask is the highest-IoU candidate;
                the decoder's dynamic_multimask_via_stability fallback only applies
                when multimask output is off, so it never fires during tracking);
              - ``"token0_logits"`` / ``"token0_mask"``: float32 / bool arrays of
                shape (H, W), the single-mask token's output — what the decoder
                would base its output on when multimask is off. Computed on every
                frame but normally discarded; raw logits, without the no-object
                masking applied to the candidates;
              - ``"token0_iou"`` / ``"token0_stability"``: floats, the predicted IoU
                and the stability score of the single-mask token's output.
                The token0 keys are only present in debug mode (``debug=True``,
                which enables the decoder's token-0 stash) and absent when the frame
                was not freshly tracked.
            These are computed for the current frame only and are not stored in memory.
        """
        if self.inference_state is None:
            raise RuntimeError("track() called before init(); initialize tracking first")

        self.frame_idx += 1
        self._last_frame = frame.copy()

        if self.debug:
            # Cleared here (in addition to the reset inside the conditioning code) so
            # a revisited already-consolidated frame, which skips inference entirely,
            # cannot report the previous frame's attention roster.
            self.predictor.last_mem_debug = None
        # Same staleness guard for the single-mask-token stash (only written when the
        # frame is freshly tracked through the mask decoder).
        self.predictor.sam_mask_decoder.last_token0_out = None

        # Run tracking on this frame
        sam_outputs = self.predictor.propagate_in_video_single(
            self.inference_state, self._last_frame, self.frame_idx,
            return_all_masks=return_all_masks,
        )
        (frame_idx, object_ids, low_res_mask, video_res_mask, obj_scores,
         all_masks_video_res, multimask_ious) = sam_outputs

        # Extract mask for our tracked object
        out_mask = np.zeros(video_res_mask[0].shape[1:], dtype=np.uint8) > 0
        obj_index = None
        for i_oid, oid in enumerate(object_ids):
            if oid == self.obj_id:
                obj_index = i_oid
                mask_logit = video_res_mask[i_oid]
                mask = (einops.rearrange(mask_logit, "1 H W -> H W") > 0).cpu().numpy()
                out_mask = np.logical_or(out_mask, mask)

        mask = out_mask

        # Trim old frames from memory to prevent unbounded growth
        trimmed = _trim_memory(
            self.predictor, frame_idx, self.inference_state["output_dict"]
        )

        self._record_debug("track", trimmed=trimmed)

        if return_all_masks:
            candidates = self._extract_candidates(
                all_masks_video_res, multimask_ious, obj_index
            )
            return mask, candidates

        return mask

    def _extract_candidates(self, all_masks_video_res, multimask_ious, obj_index):
        """Build the per-candidate dict for the tracked object (see ``track``)."""
        if all_masks_video_res is None or obj_index is None:
            # No freshly-tracked candidates available for this frame/object.
            return {
                "masks": np.empty((0, 0, 0), dtype=bool),
                "logits": np.empty((0, 0, 0), dtype=np.float32),
                "ious": np.empty((0,), dtype=np.float32),
            }
        logits = all_masks_video_res[obj_index].float().cpu().numpy()  # (M, H, W)
        ious = multimask_ious[obj_index].float().cpu().numpy()  # (M,)
        candidates = {
            "masks": logits > 0,
            "logits": logits,
            "ious": ious,
        }

        # The single-mask token's output (token 0), stashed by the mask decoder on
        # this frame's forward pass (see MaskDecoder.forward); consume and clear it.
        token0 = getattr(self.predictor.sam_mask_decoder, "last_token0_out", None)
        self.predictor.sam_mask_decoder.last_token0_out = None
        if token0 is not None:
            token0_low_res, token0_iou, token0_stability = token0
            video_hw = (
                self.inference_state["video_height"],
                self.inference_state["video_width"],
            )
            token0_video_res = torch.nn.functional.interpolate(
                token0_low_res[obj_index : obj_index + 1].float(),
                size=video_hw,
                mode="bilinear",
                align_corners=False,
            )
            token0_logits = token0_video_res[0, 0].cpu().numpy()
            candidates["token0_logits"] = token0_logits
            candidates["token0_mask"] = token0_logits > 0
            candidates["token0_iou"] = float(token0_iou[obj_index, 0])
            candidates["token0_stability"] = float(token0_stability[obj_index, 0])
        return candidates

    def correct(self, mask: np.ndarray) -> np.ndarray:
        """
        Correct the segmentation on the current (most recently tracked) frame.

        The user supplies a hand-annotated corrected mask for the frame that was
        just returned by ``track()`` (or by ``init()``). The frame itself does not
        need to be passed again: the tracker reuses the latest frame it saw, which
        guarantees the correction lands on exactly the frame that produced the mask
        being corrected. The mask is added as an authoritative conditioning frame at
        ``self.frame_idx`` so that all subsequent tracking uses it as memory. This
        mirrors the first-frame ``init()`` path (``add_new_mask_direct`` +
        ``propagate_in_video_preflight``), applied at the current frame instead of
        frame 0.

        Args:
            mask: Corrected binary segmentation mask (HxW boolean numpy array).

        Returns:
            The input mask (passed through for convenience).
        """
        if self.inference_state is None or self._last_frame is None:
            raise RuntimeError("correct() called before init(); initialize tracking first")

        # Add the corrected mask as a conditioning frame (memory encoder deferred),
        # reusing the most recently tracked frame.
        self.predictor.add_new_mask_direct(
            inference_state=self.inference_state,
            frame_idx=self.frame_idx,
            obj_id=self.obj_id,
            frame=self._last_frame,
            mask=torch.from_numpy(mask.copy()),
        )
        # Consolidate and run the memory encoder (only on this new frame). This also
        # pops the prior tracked non-conditioning output for this frame.
        self.predictor.propagate_in_video_preflight(self.inference_state)

        cleared: List[int] = []
        if self.clear_recent_memory_on_correct:
            cleared = self._clear_consolidated_non_cond_around(self.frame_idx)

        evicted: List[int] = []
        if not self.accumulate_corrections:
            evicted = self._evict_stale_corrections()

        self._record_debug("correct", cleared=cleared, evicted=evicted)

        return mask

    def _clear_consolidated_non_cond_around(self, frame_idx: int) -> List[int]:
        """
        Clear recent non-conditioning memory around a correction frame.

        The predictor's built-in ``_clear_non_cond_mem_around_input`` only clears the
        per-object output dict, which the streaming path leaves empty (it stores
        tracked outputs directly in the consolidated ``output_dict``). So we clear the
        consolidated dict that memory selection actually reads from.

        Returns:
            The frame indices that were actually removed.
        """
        r = self.predictor.memory_temporal_stride_for_eval
        n = self.predictor.num_maskmem
        non_cond = self.inference_state["output_dict"]["non_cond_frame_outputs"]
        cleared = []
        for t in range(frame_idx - r * n, frame_idx + r * n + 1):
            if non_cond.pop(t, None) is not None:
                cleared.append(t)
        return cleared

    def _evict_stale_corrections(self) -> List[int]:
        """
        Free conditioning frames that can never be selected for attention again.

        The stream only moves forward, so ``select_closest_cond_frames`` will, for any
        future frame, keep at most ``max_cond_frames_in_attn`` conditioning frames: the
        pinned first frame (when ``keep_first_cond_frame``) plus the most recent ones.
        Older corrections are unreachable forever, so we delete them (and their
        downgraded non-conditioning copies) to actually free GPU memory.

        Returns:
            The conditioning frame indices that were evicted.
        """
        N = self.predictor.max_cond_frames_in_attn
        if N == -1:
            return []  # unbounded attention; no conditioning frame is ever unreachable

        output_dict = self.inference_state["output_dict"]
        cond = output_dict["cond_frame_outputs"]
        ordered = sorted(cond.keys())
        if len(ordered) <= N:
            return []

        if self.predictor.keep_first_cond_frame:
            protected = {ordered[0]}
            if N > 1:
                protected |= set(ordered[-(N - 1):])
        else:
            protected = set(ordered[-N:])

        evicted = []
        for idx in ordered:
            if idx in protected:
                continue
            # Removes inputs + consolidated_frame_inds and downgrades the cond output
            # to non_cond (keeping mask_inputs / consolidated_frame_inds consistent for
            # the preflight assert).
            self.predictor.clear_all_points_in_frame(
                self.inference_state, idx, self.obj_id, need_output=False
            )
            # Delete the downgraded non_cond copy so the GPU memory is actually freed.
            output_dict["non_cond_frame_outputs"].pop(idx, None)
            for obj_output_dict in self.inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(idx, None)
            evicted.append(idx)
        return evicted

    def _record_debug(
        self,
        event: str,
        trimmed: Optional[List[int]] = None,
        cleared: Optional[List[int]] = None,
        evicted: Optional[List[int]] = None,
    ) -> None:
        """
        Append one debug record for the call that just finished (no-op unless
        ``debug=True``).

        All ``frame_idx`` values anywhere in the record are absolute stream indices:
        the init frame is 0 and the k-th ``track()`` call is k ("correct" records
        share the index of the last tracked frame).

        Record schema (all plain Python scalars, JSON-serializable):
          - ``event``: "init" | "track" | "correct".
          - ``frame_idx``: the current frame.
          - ``attention``: for "track" events, the memory roster actually attended
            when computing this frame (``cond_selected``/``cond_unselected``,
            ``valid_indices`` from ``frame_filter``, ``spatial_mem`` and ``obj_ptrs``
            as lists of {frame_idx, t_pos/pos, is_cond}); None for "init"/"correct"
            (mask-as-output, no memory read) and for revisited consolidated frames.
            ``t_pos`` is the temporal attention slot (0 = cond frame — shared by all
            cond frames — and 1..num_maskmem-1 for non-cond, num_maskmem-1 being the
            most recent frame); slots slide by one every step. ``pos`` feeds the
            pointer's sine temporal encoding: true frame distance for cond frames,
            recency rank for non-cond. Both lists are ordered exactly as the memory
            tokens are concatenated for cross-attention: each ``spatial_mem`` entry
            spans one memory feature map's tokens, followed by the pointer block
            where entry i occupies the (C // mem_dim) tokens starting at
            i * (C // mem_dim).
          - ``scores``: this frame's ``eff_iou_score`` / ``iou_score`` /
            ``object_score_logits`` (None when the frame has no tracked output, e.g.
            right after a correction consolidated it into a cond frame).
          - ``trimmed`` / ``cleared`` / ``evicted_corrections``: frame indices whose
            memory was deleted by this call.
          - ``mem_state``: memory-bank contents after the call — ``cond`` frame
            indices and ``non_cond_scores`` mapping frame index -> eff_iou_score.
        """
        if not self.debug:
            return
        output_dict = self.inference_state["output_dict"]
        non_cond = output_dict["non_cond_frame_outputs"]

        cur = non_cond.get(self.frame_idx)
        scores = None
        if cur is not None and "eff_iou_score" in cur:
            scores = {
                "eff_iou_score": float(cur["eff_iou_score"]),
                "iou_score": float(cur["iou_score"].flatten()[0]),
                "object_score_logits": float(cur["object_score_logits"].flatten()[0]),
            }

        attention = None
        if event == "track":
            attention = getattr(self.predictor, "last_mem_debug", None)

        self.debug_log.append(
            {
                "event": event,
                "frame_idx": self.frame_idx,
                "attention": attention,
                "scores": scores,
                "trimmed": trimmed or [],
                "cleared": cleared or [],
                "evicted_corrections": evicted or [],
                "mem_state": {
                    "cond": sorted(
                        int(t) for t in output_dict["cond_frame_outputs"]
                    ),
                    "non_cond_scores": {
                        int(t): float(out["eff_iou_score"])
                        for t, out in sorted(non_cond.items())
                        if "eff_iou_score" in out
                    },
                },
            }
        )

    def save_debug_log(self, path: str) -> None:
        """
        Save ``debug_config`` and the accumulated ``debug_log`` as JSON.

        Note that JSON turns the integer keys of ``non_cond_scores`` into strings.
        Callers doing periodic flushes on very long streams can clear
        ``self.debug_log`` after saving.
        """
        with open(path, "w") as f:
            json.dump({"config": self.debug_config, "log": self.debug_log}, f)


def _trim_memory(
        tracker: Any, frame_idx: int, output_dict: Dict[str, Any]
) -> List[int]:
    """
    Trim old frames from memory to prevent unbounded growth in long videos.

    This internal function removes frames that are no longer needed for tracking,
    keeping only the frames that the memory selection mechanism would retain.

    Args:
        tracker: The SAM3 tracker predictor instance
        frame_idx: Current frame index
        output_dict: Output dictionary from inference_state containing frame outputs

    Returns:
        The frame indices that were removed from ``non_cond_frame_outputs``.
    """
    if not tracker.use_memory_selection:
        raise NotImplementedError(
            "Memory trimming when not using memory selection not implemented yet"
        )

    memory_stride = tracker.memory_temporal_stride_for_eval

    # Keep what the tracker would keep
    selected_indices = tracker.frame_filter(
        output_dict,
        track_in_reverse=False,
        frame_idx=frame_idx,
        num_frames=frame_idx + 1,
        r=memory_stride,
    )

    # Discard all other frames
    trimmed = []
    for i in range(frame_idx - 1, 0, -memory_stride):
        if i not in selected_indices:
            # Delete only the non_cond_frame_outputs, keep the cond_ ones
            # (with direct user annotation)
            if i in output_dict["non_cond_frame_outputs"]:
                del output_dict["non_cond_frame_outputs"][i]
                trimmed.append(i)
    return trimmed
