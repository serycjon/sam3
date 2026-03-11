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

    def __init__(self) -> None:
        """
        Initialize the streaming tracker with SAM3 model.
        
        Loads the SAM3 video model and sets up the tracking predictor.
        """
        from sam3.model_builder import build_sam3_video_model

        sam3_model = build_sam3_video_model()
        predictor = sam3_model.tracker
        predictor.backbone = sam3_model.detector.backbone

        self.predictor = predictor
        self.obj_id = 1

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
        # This is a workaround for the video model expecting a frame count
        DUMMY_N_FRAMES = 13
        
        self.inference_state = self.predictor.init_state(
            video_height=frame.shape[0],
            video_width=frame.shape[1],
            num_frames=DUMMY_N_FRAMES,
        )
        self.frame_idx = 0
        
        # Clear any previous state
        self.predictor.clear_all_points_in_video(self.inference_state)
        
        # Add the initial mask
        self.predictor.add_new_mask_direct(
            inference_state=self.inference_state,
            frame_idx=self.frame_idx,
            obj_id=self.obj_id,
            frame=frame.copy(),
            mask=torch.from_numpy(mask.copy()),
        )
        
        # Prepare for tracking
        self.predictor.propagate_in_video_preflight(self.inference_state)

        return mask

    def track(self, frame: np.ndarray) -> np.ndarray:
        """
        Track the object in the next frame.
        
        Args:
            frame: OpenCV frame (HxWx3 numpy array in uint8 BGR format)
        
        Returns:
            Binary segmentation mask (HxW boolean numpy array)
            True/1 indicates the object, False/0 indicates background
        """
        self.frame_idx += 1
        
        # Run tracking on this frame
        sam_outputs = self.predictor.propagate_in_video_single(
            self.inference_state, frame.copy(), self.frame_idx
        )
        frame_idx, object_ids, low_res_mask, video_res_mask, obj_scores = sam_outputs

        # Extract mask for our tracked object
        out_mask = np.zeros(video_res_mask[0].shape[1:], dtype=np.uint8) > 0
        for i_oid, oid in enumerate(object_ids):
            if oid == self.obj_id:
                mask_logit = video_res_mask[i_oid]
                mask = (einops.rearrange(mask_logit, "1 H W -> H W") > 0).cpu().numpy()
                out_mask = np.logical_or(out_mask, mask)

        mask = out_mask

        # Trim old frames from memory to prevent unbounded growth
        _trim_memory(self.predictor, frame_idx, self.inference_state["output_dict"])

        return mask


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
