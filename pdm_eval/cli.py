"""Command-line entry points for the MetroPT PdM framework."""

from __future__ import annotations

from . import pipeline
from .logging_utils import log_to_file


def main() -> None:
    """Run the configured MetroPT evaluation pipeline with file logging."""
    with log_to_file(
        pipeline._effective_detector_type(pipeline.EXPERIMENT_MODE),
        pipeline._resolve_artifact_mode(pipeline.EXPERIMENT_MODE),
        artifacts_root=pipeline.ARTIFACTS_ROOT,
    ):
        pipeline.main()


if __name__ == "__main__":
    main()

