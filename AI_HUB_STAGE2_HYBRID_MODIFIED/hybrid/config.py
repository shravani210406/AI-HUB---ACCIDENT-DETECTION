from pathlib import Path
import hashlib
import json
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path=None):
    """Load one complete configuration; resolve artifact paths against project root."""
    with open(path or ROOT / 'configs/default.yaml', encoding='utf-8') as f:
        c = yaml.safe_load(f)
    for key in ('target_process_fps', 'window_seconds', 'stride_seconds'):
        if c[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not 0 <= c['accident_threshold_low'] < c['accident_threshold_high'] <= 1:
        raise ValueError('Require 0 <= low < high <= 1')
    if not 1 <= c['vote_min_positive'] <= c['vote_window']:
        raise ValueError('Invalid K-of-N voting')
    if c['clip_frame_interval']<1 or c['exit_steps']<1:
        raise ValueError('CLIP interval and exit steps must be positive integers')
    if not 0<c['motion_ema']<=1:
        raise ValueError('motion_ema must be in (0,1]')
    for key in ('merge_gap_seconds','lookback_seconds','minimum_event_duration'):
        if c[key]<0:raise ValueError(f'{key} must be nonnegative')
    return c

def resolve(path):
    p = Path(str(path).replace('\\','/'))
    return p if p.is_absolute() else ROOT / p

def feature_signature(c):
    keys = ('yolo_model', 'input_size', 'target_process_fps', 'yolo_confidence',
            'nms_iou', 'tracker_type', 'clip_model', 'clip_pretrained',
            'clip_frame_interval', 'motion_ema', 'track_max_gap', 'camera_compensation')
    return hashlib.sha256(json.dumps({k:c[k] for k in keys}, sort_keys=True).encode()).hexdigest()
