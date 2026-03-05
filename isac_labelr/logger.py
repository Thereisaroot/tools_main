from __future__ import annotations

import logging
from pathlib import Path


def build_run_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"isac_labelr.run.{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.handlers.clear()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )

    logger.addHandler(file_handler)
    return logger
