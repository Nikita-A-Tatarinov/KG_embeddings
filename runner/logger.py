# train/logger.py
from __future__ import annotations

import csv
import os
import time
from typing import Any


class CSVLogger:
    def __init__(self, out_dir: str, filename: str = "log.csv", fieldnames=None):
        self.path = os.path.join(out_dir, filename)
        self.file = open(self.path, "w", newline="", encoding="utf-8")
        self.fieldnames = list(fieldnames) if fieldnames is not None else None
        self.writer = None

    def log(self, row: dict[str, Any]):
        if self.writer is None:
            # If no fieldnames provided, infer from the first row
            if self.fieldnames is None:
                self.fieldnames = list(row.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
        # Normalize: fill any missing columns with empty values
        full_row = {k: row.get(k, "") for k in self.fieldnames}
        self.writer.writerow(full_row)
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
