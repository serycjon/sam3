# Copyright (c) 2026 Jonas Serych
"""
Streaming SAM3 tracker for long videos and video streams.

This module provides a high-level interface for single-object tracking in video streams,
optimized for memory efficiency and suitable for long-running tracking tasks.
"""

from typing import Any, Dict

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
        self.frame_idx = 0
        self._last_frame = frame.copy()

        # Clear any previous state
        self.predictor.clear_all_points_in_video(self.inference_state)

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
                candidate (the best of which is what the default mask uses).
            These are computed for the current frame only and are not stored in memory.
        """
        self.frame_idx += 1
        self._last_frame = frame.copy()

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
        _trim_memory(self.predictor, frame_idx, self.inference_state["output_dict"])

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
        return {
            "masks": logits > 0,
            "logits": logits,
            "ious": ious,
        }

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

        if self.clear_recent_memory_on_correct:
            self._clear_consolidated_non_cond_around(self.frame_idx)

        if not self.accumulate_corrections:
            self._evict_stale_corrections()

        return mask

    def _clear_consolidated_non_cond_around(self, frame_idx: int) -> None:
        """
        Clear recent non-conditioning memory around a correction frame.

        The predictor's built-in ``_clear_non_cond_mem_around_input`` only clears the
        per-object output dict, which the streaming path leaves empty (it stores
        tracked outputs directly in the consolidated ``output_dict``). So we clear the
        consolidated dict that memory selection actually reads from.
        """
        r = self.predictor.memory_temporal_stride_for_eval
        n = self.predictor.num_maskmem
        non_cond = self.inference_state["output_dict"]["non_cond_frame_outputs"]
        for t in range(frame_idx - r * n, frame_idx + r * n + 1):
            non_cond.pop(t, None)

    def _evict_stale_corrections(self) -> None:
        """
        Free conditioning frames that can never be selected for attention again.

        The stream only moves forward, so ``select_closest_cond_frames`` will, for any
        future frame, keep at most ``max_cond_frames_in_attn`` conditioning frames: the
        pinned first frame (when ``keep_first_cond_frame``) plus the most recent ones.
        Older corrections are unreachable forever, so we delete them (and their
        downgraded non-conditioning copies) to actually free GPU memory.
        """
        N = self.predictor.max_cond_frames_in_attn
        if N == -1:
            return  # unbounded attention; no conditioning frame is ever unreachable

        output_dict = self.inference_state["output_dict"]
        cond = output_dict["cond_frame_outputs"]
        ordered = sorted(cond.keys())
        if len(ordered) <= N:
            return

        if self.predictor.keep_first_cond_frame:
            protected = {ordered[0]}
            if N > 1:
                protected |= set(ordered[-(N - 1):])
        else:
            protected = set(ordered[-N:])

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


def _trim_memory(
        tracker: Any, frame_idx: int, output_dict: Dict[str, Any]
) -> None:
    """
    Trim old frames from memory to prevent unbounded growth in long videos.
    
    This internal function removes frames that are no longer needed for tracking,
    keeping only the frames that the memory selection mechanism would retain.
    
    Args:
        tracker: The SAM3 tracker predictor instance
        frame_idx: Current frame index
        output_dict: Output dictionary from inference_state containing frame outputs
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
    for i in range(frame_idx - 1, 0, -memory_stride):
        if i not in selected_indices:
            # Delete only the non_cond_frame_outputs, keep the cond_ ones
            # (with direct user annotation)
            if i in output_dict["non_cond_frame_outputs"]:
                del output_dict["non_cond_frame_outputs"][i]
