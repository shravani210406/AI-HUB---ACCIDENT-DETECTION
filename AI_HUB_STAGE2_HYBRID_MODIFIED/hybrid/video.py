"""PTS-based decoding. Processing clocks are used only for performance measurements."""
from dataclasses import dataclass
import logging
import math

def timestamp(pts, time_base, index: int, fps: float | None) -> float:
    if pts is not None and time_base is not None:
        value = float(pts * time_base)
        if math.isfinite(value):
            return value
    if fps is None or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Neither usable presentation timestamp nor trustworthy frame rate')
    return index / fps

def format_time(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Timestamp must be finite and nonnegative')
    total = int(math.floor(seconds * 1000 + 0.5))
    s, ms = divmod(total, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f'{h:02}:{m:02}:{s:02}.{ms:03}'

@dataclass
class VideoInfo:
    fps: float | None = None
    duration: float = 0.0
    decoded: int = 0
    sampled: int = 0
    pts_origin: float | None = None
    fallback_frames: int = 0

class VideoReader:
    def __init__(self, path, sample_fps: float):
        if sample_fps <= 0:
            raise ValueError('sample_fps must be positive')
        self.path, self.sample_fps = path, sample_fps
        self.info = VideoInfo()

    def __iter__(self):
        import av
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            self.info.fps = float(stream.average_rate) if stream.average_rate else None
            logging.info('Video loaded: %s; source average FPS=%s',self.path,self.info.fps)
            fallback_fps = self.info.fps if stream.base_rate == stream.average_rate else None
            next_sample, previous = 0.0, -1.0
            for index, frame in enumerate(container.decode(stream)):
                valid_pts = frame.pts is not None and frame.time_base is not None
                raw = timestamp(frame.pts, frame.time_base, index, fallback_fps)
                if self.info.pts_origin is None:
                    self.info.pts_origin = raw if valid_pts else 0.0
                t = raw - self.info.pts_origin if valid_pts else raw
                self.info.fallback_frames += int(not valid_pts)
                if t < previous - 1e-6:
                    raise ValueError('Nonmonotonic video timestamps; repair input before inference')
                previous = t
                self.info.decoded += 1
                duration = float(frame.duration * frame.time_base) if frame.duration and frame.time_base else (1 / self.info.fps if self.info.fps else 0)
                self.info.duration = max(self.info.duration, t + duration)
                if t + 1e-8 >= next_sample:
                    self.info.sampled += 1
                    next_sample = (math.floor(t * self.sample_fps + 1e-8) + 1) / self.sample_fps
                    yield t, frame.to_ndarray(format='bgr24')
            if not self.info.decoded:
                raise ValueError('Video has no decodable frames')
        logging.info('Decoded %s: %s', self.path, self.info)
