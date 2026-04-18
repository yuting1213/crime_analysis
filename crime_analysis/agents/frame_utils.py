"""Shared frame-sampling helpers used by both Planner and VLM diagnostics.

Keeping a single implementation prevents drift between `pipeline._vlm_classify`
and `scripts.test_vlm_classify` — see the Arrest043 order-bias incident for why
that matters.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def uniform_keyframes(video_path: str, n: int = 8) -> List:
    """
    Uniformly sample *n* frames from *video_path*.

    Returns
    -------
    list[PIL.Image.Image]
        RGB PIL frames; empty list if the video cannot be read or has no
        decodable frames.
    """
    import cv2
    from PIL import Image as PILImage

    if not video_path or not Path(video_path).exists():
        return []

    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0 or n <= 0:
            return []
        indices = [int(i * total / n) for i in range(n)]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(
                    PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                )
        return frames
    finally:
        cap.release()


def fallback_from_frame_list(frames, n: int = 8) -> List:
    """
    When the original video file is unavailable, sample *n* PIL frames
    uniformly from an already-loaded frame list (numpy arrays or PIL Images).
    """
    import numpy as np
    from PIL import Image as PILImage

    if not frames or n <= 0:
        return []

    total = len(frames)
    indices = [int(i * total / n) for i in range(n)] if total > n else list(range(total))
    out = []
    for idx in indices:
        f = frames[idx]
        if isinstance(f, PILImage.Image):
            out.append(f)
        elif isinstance(f, np.ndarray):
            out.append(PILImage.fromarray(f))
    return out
