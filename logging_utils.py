#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple log tee helper to save latest run by detector group + regime."""

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
    artifacts_root: Optional[str] = "artifacts",
) -> Iterator[Path]:
    detector = str(detector_type).lower().strip()
    if detector in {"iforest", "isolation_forest", "isolationforest"}:
        detector_group = "iforest"
    elif detector in {"autoencoder", "ae", "nianetvae", "nianetvae_pretrained"}:
        detector_group = "vae"
    else:
        detector_group = detector or "unknown"

    mode = str(experiment_mode).lower().strip()
    if mode in {"per_maint", "per-maint", "permaint"}:
        mode = "per_maint"
    elif mode == "single":
        mode = "single"
    else:
        mode = mode or "unknown"

    log_dir = Path(artifacts_root or "artifacts") / detector_group / mode / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

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
