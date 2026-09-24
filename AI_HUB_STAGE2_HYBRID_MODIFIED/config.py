from pathlib import Path


# --------------------------------------------------
# PROJECT
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------
# MODEL — Hybrid runtime artifacts
# --------------------------------------------------

MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n_openvino_model"
MODEL_NAME = "YOLO11n + ByteTrack + Motion + CLIP + TCN"

DETECTION_ENGINE = "YOLO11n_ByteTrack"
ACCIDENT_ALGORITHM = "Hybrid_CLIP_TCN"

# The complete trained Hybrid model configuration. The YAML contains the
# detector, CLIP, temporal-checkpoint and calibrated decision parameters.
HYBRID_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


# --------------------------------------------------
# VIDEO SAMPLING
# --------------------------------------------------

SAMPLE_FPS = 10


# --------------------------------------------------
# RESOLUTION
# --------------------------------------------------

RESIZE_ENABLED = True
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 640


# --------------------------------------------------
# IMAGE ENHANCEMENT
# --------------------------------------------------

ENHANCEMENT_ENABLED = False
BRIGHTNESS = 0
CONTRAST = 1.0


# --------------------------------------------------
# DETECTION
# --------------------------------------------------

CONFIDENCE_THRESHOLD = 0.50
DEVICE = "cpu"


# --------------------------------------------------
# PYTORCH CPU SETTINGS
# --------------------------------------------------

TORCH_NUM_THREADS = None
TORCH_NUM_INTEROP_THREADS = 1


# --------------------------------------------------
# OUTPUT / LOGGING
# --------------------------------------------------

VERBOSE = False
