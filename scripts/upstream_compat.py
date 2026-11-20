"""Runtime compatibility patches for pinned upstream UmeTrack (third_party/ is never edited).

VideoStream.__iter__ (lib/tracker/video_pose_data.py) logs `int(stream.average_rate)`, which
fails on PyAV >= 19 where average_rate is an AVRational. The replacement below is upstream's
body with only that log line changed to float(); frame decoding is unchanged. `apply()`
verifies that claim against upstream's source before patching.
"""

import inspect
import textwrap

import av
import numpy as np

ORIGINAL_LOG = 'print(f"Opened ({int(stream.average_rate)} fps) video from {self._data_path}")'
PATCHED_LOG = 'print(f"Opened ({float(stream.average_rate):g} fps) video from {self._data_path}")'

APPLIED = []


def _video_stream_iter(self):
    container = av.open(self._data_path)
    # take first video stream
    stream = container.streams.video[0]
    print(f"Opened ({float(stream.average_rate):g} fps) video from {self._data_path}")

    for idx, frame in enumerate(container.decode(stream)):
        raw_mono_image_np = np.array(frame.to_image())[..., 0]
        yield raw_mono_image_np


def _body(fn) -> str:
    src = textwrap.dedent(inspect.getsource(fn))
    lines = [ln.strip() for ln in src.splitlines()[1:]]  # drop the def line
    return "\n".join(ln for ln in lines if ln)


def apply() -> list:
    """Patch upstream VideoStream.__iter__; returns descriptions of applied patches."""
    from lib.tracker.video_pose_data import VideoStream

    if VideoStream.__iter__ is _video_stream_iter:
        return APPLIED
    expected = _body(_video_stream_iter).replace(PATCHED_LOG, ORIGINAL_LOG)
    if _body(VideoStream.__iter__) != expected:
        raise RuntimeError("upstream VideoStream.__iter__ changed; refusing to patch")
    VideoStream.__iter__ = _video_stream_iter
    APPLIED.append("lib.tracker.video_pose_data.VideoStream.__iter__: fps log line uses float() (PyAV>=19)")
    return APPLIED
