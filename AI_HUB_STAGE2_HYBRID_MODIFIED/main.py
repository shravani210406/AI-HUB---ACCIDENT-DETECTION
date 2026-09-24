"""AI Hub Stage 2 experiment runner, using the trained Hybrid model by default."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import yaml

from config import (
    ACCIDENT_ALGORITHM,
    DETECTION_ENGINE,
    HYBRID_CONFIG_PATH,
    PROJECT_ROOT,
    TORCH_NUM_INTEROP_THREADS,
    TORCH_NUM_THREADS,
)
from experiment_config import build_experiment_name, get_experiment_config
from hybrid.config import load_config
from hybrid.pipeline import HybridPipeline
from postprocessing.log_reader import LogReader
from postprocessing.metrics import MetricsCalculator
from postprocessing.report_generator import ReportGenerator
from result_logging.result_logger import ResultLogger


# ============================================================
# PROJECT PATHS — kept in the original Stage 2 layout
# ============================================================

DATASET_ROOT = PROJECT_ROOT / "new_dataset_final"
ACCIDENT_FOLDER = DATASET_ROOT / "new_acc"
NORMAL_FOLDER = DATASET_ROOT / "new_norm"
EXPERIMENT_FILE = PROJECT_ROOT / "experiment_videos.txt"

EXPERIMENT_BASE_NAME = build_experiment_name()
EXCEL_FILENAME = "complete_results.xlsx"
CSV_FILENAME = "video_results.csv"
CONFIG_FILENAME = "config.json"


def get_next_experiment_name(base_name: str) -> str:
    """Use Stage 2's numbered experiment folders without overwriting a run."""
    logs_root = PROJECT_ROOT / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    highest_number = 0
    prefix = f"{base_name}_"
    for folder in logs_root.iterdir():
        if folder.is_dir() and folder.name.startswith(prefix):
            suffix = folder.name[len(prefix):]
            if suffix.isdigit():
                highest_number = max(highest_number, int(suffix))
    return f"{base_name}_{highest_number + 1:03d}"


def configure_cpu() -> None:
    if TORCH_NUM_THREADS is not None:
        torch.set_num_threads(TORCH_NUM_THREADS)
    if TORCH_NUM_INTEROP_THREADS is not None:
        torch.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)
    print("\nCPU configuration:")
    print(f"  PyTorch threads: {torch.get_num_threads()}")
    print(f"  PyTorch inter-op threads: {torch.get_num_interop_threads()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")


def read_experiment_videos() -> list[dict[str, str]]:
    """Read the unchanged Stage 2 ``video_name,label`` experiment format."""
    if not EXPERIMENT_FILE.exists():
        raise FileNotFoundError(f"Experiment file not found: {EXPERIMENT_FILE}")
    experiments: list[dict[str, str]] = []
    with EXPERIMENT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [value.strip() for value in line.split(",")]
            if len(parts) != 2 or parts[1].lower() not in {"accident", "normal"}:
                raise ValueError(
                    f"Invalid experiment_videos.txt line {line_number}: {line}. "
                    "Expected: video_name,accident or video_name,normal"
                )
            experiments.append({"video_name": parts[0], "ground_truth": parts[1].lower()})
    if not experiments:
        raise ValueError("experiment_videos.txt contains no videos")
    return experiments


def find_video(video_name: str, ground_truth: str) -> Path:
    """Find the exact video listed in experiment_videos.txt.

    ``experiment_videos.txt`` is the batch controller. A listed filename may
    be in any folder within the project; the original Stage 2 new_acc/new_norm
    locations are merely checked first for compatibility. A full path in the
    list is also accepted.
    """
    listed_path = Path(video_name)
    direct_candidates = [
        listed_path,
        PROJECT_ROOT / listed_path,
        (ACCIDENT_FOLDER if ground_truth == "accident" else NORMAL_FOLDER) / listed_path,
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = sorted(
        path for path in PROJECT_ROOT.rglob(listed_path.name)
        if path.is_file() and path.name.lower() == listed_path.name.lower()
    )
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise FileNotFoundError(
            f"More than one video named '{video_name}' was found in {DATASET_ROOT}. "
            "Use a path relative to this project or a full path in experiment_videos.txt."
        )
    raise FileNotFoundError(
        f"Video listed in experiment_videos.txt was not found: {video_name}\n"
        f"Searched every folder under: {PROJECT_ROOT}\n"
        "Put the video anywhere inside this project, or use its full path in experiment_videos.txt."
    )


def prepare_log_directory(log_directory: Path) -> None:
    """The experiment name is new, so no previous Stage 2 result is overwritten."""
    log_directory.mkdir(parents=True, exist_ok=True)
    print(f"\nExperiment log directory:\n  {log_directory}")


def save_hybrid_diagnostics(result: dict, diagnostics_directory: Path, video_path: Path) -> Path:
    diagnostics_directory.mkdir(parents=True, exist_ok=True)
    output_path = diagnostics_directory / f"{video_path.stem}.json"
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return output_path


def process_video(
    video_path: Path,
    ground_truth: str,
    hybrid_pipeline: HybridPipeline,
    result_logger: ResultLogger,
    diagnostics_directory: Path,
) -> bool:
    """Run one complete video through Hybrid, then preserve the Stage 2 log shape."""
    print("\n" + "=" * 70)
    print(f"Processing: {video_path.name}")
    print(f"Ground truth: {ground_truth}")
    print("=" * 70)
    try:
        hybrid_result = hybrid_pipeline.run(video_path)
        events = hybrid_result["events"]
        prediction = "accident" if hybrid_result["prediction"] == "ACCIDENT" else "normal"
        first_accident_timestamp = events[0]["onset_seconds"] if events else None
        first_accident_alert_frame = (
            int(round(events[0]["confirmation_seconds"] * hybrid_result["source_fps"])) + 1
            if events and hybrid_result.get("source_fps") else None
        )
        suspicious_frame_count = sum(step["accident_state"] for step in hybrid_result["trace"])

        print("\nResult:")
        print(f"  Prediction: {prediction}")
        print(f"  First accident timestamp: {first_accident_timestamp}")
        print(f"  First accident alert frame: {first_accident_alert_frame}")
        print(f"  Frames processed: {hybrid_result['sampled']}")
        print(f"  Total detections: {hybrid_result['detection_count']}")
        print(f"  Vehicle detections: {hybrid_result['vehicle_detection_count']}")
        print(f"  Hybrid confidence: {hybrid_result['confidence']:.4f}")
        print(f"  Processing time: {hybrid_result['processing_seconds']:.2f} seconds")

        log_path = result_logger.save_result(
            video_name=video_path.name,
            ground_truth=ground_truth,
            prediction=prediction,
            first_accident_timestamp=first_accident_timestamp,
            frames_processed=hybrid_result["sampled"],
            detection_count=hybrid_result["detection_count"],
            vehicle_detection_count=hybrid_result["vehicle_detection_count"],
            suspicious_frame_count=suspicious_frame_count,
            processing_time=hybrid_result["processing_seconds"],
        )
        diagnostics_path = save_hybrid_diagnostics(hybrid_result, diagnostics_directory, video_path)
        print(f"  Stage 2 log saved: {log_path}")
        print(f"  Hybrid diagnostics saved: {diagnostics_path}")
        return True
    except Exception as error:
        print("\nERROR while processing:")
        print(f"  Video: {video_path.name}")
        print(f"  Error: {error}")
        return False


def main() -> None:
    experiment_name = get_next_experiment_name(EXPERIMENT_BASE_NAME)
    log_directory = PROJECT_ROOT / "logs" / experiment_name
    report_directory = PROJECT_ROOT / "reports" / experiment_name
    config_directory = PROJECT_ROOT / "experiment_configurations" / experiment_name
    diagnostics_directory = log_directory / "hybrid_diagnostics"

    print("\n" + "=" * 70)
    print("AI HUB — HYBRID ACCIDENT DETECTION EXPERIMENT")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Detection engine: {DETECTION_ENGINE}")
    print(f"Accident algorithm: {ACCIDENT_ALGORITHM}")

    configure_cpu()
    prepare_log_directory(log_directory)
    report_directory.mkdir(parents=True, exist_ok=True)
    config_directory.mkdir(parents=True, exist_ok=True)

    hybrid_config = load_config(HYBRID_CONFIG_PATH)
    config_snapshot = {**get_experiment_config(), "hybrid_config": hybrid_config}
    (config_directory / CONFIG_FILENAME).write_text(json.dumps(config_snapshot, indent=4), encoding="utf-8")
    (config_directory / "hybrid_config.yaml").write_text(yaml.safe_dump(hybrid_config, sort_keys=False), encoding="utf-8")

    experiments = read_experiment_videos()
    accident_count = sum(item["ground_truth"] == "accident" for item in experiments)
    print("\nExperiment configuration:")
    print(f"  Total videos: {len(experiments)}")
    print(f"  Accident videos: {accident_count}")
    print(f"  Normal videos: {len(experiments) - accident_count}")

    print("\nInitializing Hybrid model pipeline...")
    hybrid_pipeline = HybridPipeline(hybrid_config)
    result_logger = ResultLogger(log_directory)

    successful_videos = 0
    failed_videos: list[str] = []
    experiment_start = time.perf_counter()
    for index, experiment in enumerate(experiments, start=1):
        print(f"\n\n[{index}/{len(experiments)}]")
        video_name, ground_truth = experiment["video_name"], experiment["ground_truth"]
        try:
            video_path = find_video(video_name, ground_truth)
        except Exception as error:
            print(f"\nERROR: {error}")
            failed_videos.append(video_name)
            continue
        if process_video(video_path, ground_truth, hybrid_pipeline, result_logger, diagnostics_directory):
            successful_videos += 1
        else:
            failed_videos.append(video_name)

    experiment_time = time.perf_counter() - experiment_start
    print("\n\n" + "=" * 70)
    print("VIDEO PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Successful videos: {successful_videos}/{len(experiments)}")
    print(f"Total experiment time: {experiment_time:.2f} seconds")
    if failed_videos:
        print("\nFinal metrics and report generation stopped because these videos failed:")
        for video_name in failed_videos:
            print(f"  - {video_name}")
        return

    results = LogReader(log_directory).read_all_logs()
    if len(results) != len(experiments):
        print(f"\nERROR: Expected {len(experiments)} result logs, found {len(results)}. Report generation stopped.")
        return
    metrics = MetricsCalculator().calculate_metrics(results)
    print("\n" + "=" * 70)
    print("HYBRID EXPERIMENT RESULTS")
    print("=" * 70)
    for name in ("TP", "TN", "FP", "FN"):
        print(f"{name}:          {metrics[name]}")
    for name in ("Accuracy", "Precision", "Recall", "F1", "Specificity"):
        print(f"{name}: {metrics[name] * 100:.2f}%")

    report_generator = ReportGenerator(report_directory)
    excel_path = report_generator.generate_excel(results, metrics, filename=EXCEL_FILENAME)
    csv_path = report_generator.generate_csv(results, filename=CSV_FILENAME)
    print("\n" + "=" * 70)
    print("EXPERIMENT FINISHED SUCCESSFULLY")
    print("=" * 70)
    print(f"Logs:  {log_directory}")
    print(f"Excel: {excel_path}")
    print(f"CSV:   {csv_path}")


if __name__ == "__main__":
    main()
