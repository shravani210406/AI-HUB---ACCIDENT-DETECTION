from config import (
    MODEL_NAME,
    DETECTION_ENGINE,
    ACCIDENT_ALGORITHM,
    SAMPLE_FPS,
    RESIZE_ENABLED,
    RESIZE_WIDTH,
    RESIZE_HEIGHT,
    ENHANCEMENT_ENABLED,
    BRIGHTNESS,
    CONTRAST,
)


def build_experiment_name():
    """Build a descriptive experiment folder name from config.py."""

    parts = [
        ACCIDENT_ALGORITHM,
        DETECTION_ENGINE,
        f"{SAMPLE_FPS}fps",
    ]

    if RESIZE_ENABLED:
        parts.append(f"{RESIZE_WIDTH}x{RESIZE_HEIGHT}")
    else:
        parts.append("original_size")

    if ENHANCEMENT_ENABLED:
        parts.append("enhanced")
    else:
        parts.append("no_enhancement")

    return "_".join(parts)


def get_experiment_config():
    """Return the configuration values used for this experiment."""

    return {
        "model_name": MODEL_NAME,
        "detection_engine": DETECTION_ENGINE,
        "accident_algorithm": ACCIDENT_ALGORITHM,
        "sample_fps": SAMPLE_FPS,
        "resize_enabled": RESIZE_ENABLED,
        "resize_width": RESIZE_WIDTH,
        "resize_height": RESIZE_HEIGHT,
        "enhancement_enabled": ENHANCEMENT_ENABLED,
        "brightness": BRIGHTNESS,
        "contrast": CONTRAST,
    }
