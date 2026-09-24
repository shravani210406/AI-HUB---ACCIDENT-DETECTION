"""Connect web uploads to the bundled Stage 2 Hybrid accident detector."""

from pathlib import Path
import sys
import threading
import uuid

import cv2


WEB_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = WEB_ROOT / "AI_HUB_STAGE2_HYBRID_MODIFIED"
SNAPSHOT_DIR = WEB_ROOT / "uploads" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

_pipeline = None
_pipeline_lock = threading.Lock()


def _save_snapshot(video_path: str, seconds: float) -> str | None:
    """Keep the existing result-page snapshot, using the model's event time."""
    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000)
        ok, frame = capture.read()
        if not ok:
            return None
        filename = f"snapshot_{uuid.uuid4().hex[:8]}.jpg"
        if cv2.imwrite(str(SNAPSHOT_DIR / filename), frame):
            return filename
        return None
    finally:
        capture.release()


def detect_accident(video_path: str) -> dict:
    """Run the supplied Hybrid model and return the web page's existing fields."""
    global _pipeline

    if not BACKEND_ROOT.is_dir():
        raise FileNotFoundError(f"Final backend is missing: {BACKEND_ROOT}")

    with _pipeline_lock:
        if _pipeline is None:
            sys.path.insert(0, str(BACKEND_ROOT))
            from hybrid.config import load_config
            from hybrid.pipeline import HybridPipeline

            _pipeline = HybridPipeline(load_config())
        result = _pipeline.run(Path(video_path))

    if result.get("status") != "OK":
        raise RuntimeError(f"Hybrid detector status: {result.get('status')}")

    prediction = result["prediction"]
    if prediction not in {"ACCIDENT", "NORMAL"}:
        raise ValueError(f"Unexpected Hybrid prediction: {prediction}")

    accident = prediction == "ACCIDENT"
    events = result["events"]
    if accident and not events:
        raise ValueError("Hybrid predicted an accident without an event timestamp")

    if not accident:
        return {"accident": False, "confidence": None, "timestamp": None, "snapshot": None}

    event = events[0]
    onset_seconds = float(event["onset_seconds"])
    timestamp = event["onset_timestamp"]
    confidence = round(float(event["confidence"]) * 100, 2)
    snapshot = _save_snapshot(video_path, onset_seconds)
    return {
        "accident": True,
        "confidence": confidence,
        "timestamp": timestamp,
        "snapshot": snapshot,
    }
