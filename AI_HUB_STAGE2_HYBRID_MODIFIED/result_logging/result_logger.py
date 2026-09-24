import json
from pathlib import Path


class ResultLogger:
    """
    Saves the result of one processed video.

    Each video gets its own JSON log file.
    These raw logs are later read by the post-processing
    modules to create the consolidated CSV and Excel reports.
    """

    def __init__(self, log_directory):
        self.log_directory = Path(log_directory)

        self.log_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_result(
        self,
        video_name,
        ground_truth,
        prediction,
        first_accident_timestamp,
        frames_processed,
        detection_count,
        vehicle_detection_count,
        suspicious_frame_count,
        processing_time
    ):
        result = {
            "video_name": video_name,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "first_accident_timestamp": (
                first_accident_timestamp
            ),
            "frames_processed": frames_processed,
            "detection_count": detection_count,
            "vehicle_detection_count": (
                vehicle_detection_count
            ),
            "suspicious_frame_count": (
                suspicious_frame_count
            ),
            "processing_time_seconds": (
                processing_time
            )
        }

        output_path = (
            self.log_directory
            / f"{Path(video_name).stem}.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                result,
                file,
                indent=4
            )

        return output_path