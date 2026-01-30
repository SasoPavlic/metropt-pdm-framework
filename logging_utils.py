#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple log tee helper to save latest run per detector + mode.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for s in self._streams:
            s.write(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


@contextmanager
def log_to_file(
    detector_type: str,
    experiment_mode: str,
    logs_dir: Optional[str] = "logs",
) -> Iterator[Path]:
    log_dir = Path(logs_dir or "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    detector = str(detector_type).lower().strip()
    mode = str(experiment_mode).lower().strip()
    log_path = log_dir / f"{detector}_{mode}.log"

    orig_out, orig_err = sys.stdout, sys.stderr
    f = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(orig_out, f)
    sys.stderr = _Tee(orig_err, f)
    try:
        print(f"[INFO] Log file: {log_path}")
        yield log_path
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err
        f.close()
