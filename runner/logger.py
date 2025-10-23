# train/logger.py
from __future__ import annotations

import csv
import os
import time
from typing import Any


class CSVLogger:
    def __init__(self, out_dir: str, filename: str = "log.csv"):
        self.path = os.path.join(out_dir, filename)
        self.file = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = None

    def log(self, row: dict[str, Any]):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row.keys()))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0
