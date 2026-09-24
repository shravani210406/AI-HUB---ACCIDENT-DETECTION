import json
from pathlib import Path


class LogReader:
    """
    Reads individual JSON result logs created by ResultLogger.
    """

    def __init__(self, log_directory):
        self.log_directory = Path(log_directory)

        if not self.log_directory.exists():
            raise FileNotFoundError(
                f"Log directory not found:\n{self.log_directory}"
            )

    def read_log(self, log_path):
        """
        Read one JSON log file.
        """
        log_path = Path(log_path)

        with open(
            log_path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    def read_all_logs(self):
        """
        Read all JSON result logs in the log directory.

        Returns:
            A list of dictionaries, one for each video.
        """
        log_files = sorted(
            self.log_directory.glob("*.json")
        )

        results = []

        for log_file in log_files:
            result = self.read_log(log_file)
            results.append(result)

        return results