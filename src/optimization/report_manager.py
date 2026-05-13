from pathlib import Path
from datetime import datetime
import shutil


class ReportManager:

    def __init__(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.history_dir = (
            Path("reports/history") / f"run_{timestamp}"
        )

        self.latest_dir = Path("reports/latest")

        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.latest_dir.mkdir(parents=True, exist_ok=True)

    def archive(self, file_path):

        file_path = Path(file_path)

        if not file_path.exists():
            return

        history_target = self.history_dir / file_path.name
        latest_target = self.latest_dir / file_path.name

        shutil.copy(file_path, history_target)
        shutil.copy(file_path, latest_target)

        print(f"[ARCHIVED] {file_path.name}")