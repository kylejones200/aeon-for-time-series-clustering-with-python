#!/usr/bin/env python3
"""Aeon for time series clustering with Python.

CLI entry point: synthetic three-class data, KNN time-series classifier, metrics, plot.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.core import (
    evaluate_classifier,
    fit_classifier,
    generate_dataset,
    plot_sample_series,
    split_data,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Labels for aeon.datasets.make_example_3_class_dataset (three synthetic shapes).
SAMPLE_CLASS_LABELS = ("Sine", "Cosine", "Sine2x")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for CLI runs."""
    logging.basicConfig(level=level, format=LOG_FORMAT)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load pipeline settings from YAML.

    Args:
        config_path: Path to config file. Defaults to ``config.yaml`` beside this module.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (for tests). Uses ``sys.argv`` when omitted.

    Returns:
        Parsed namespace with ``config`` and ``output_dir`` paths.
    """
    parser = argparse.ArgumentParser(description="Time series clustering with aeon")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for figures (overrides config output.figures_dir)",
    )
    return parser.parse_args(argv)


def resolve_output_dir(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    """Choose figure output directory from CLI or config."""
    if args.output_dir is not None:
        return args.output_dir
    return Path(config["output"]["figures_dir"])


def run_pipeline(
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate data, train classifier, evaluate, and write figures.

    Args:
        config: Loaded YAML config with ``data``, ``model``, and ``output`` sections.
        output_dir: Directory for saved plots.

    Returns:
        Feature array, label array, and evaluation metrics from ``evaluate_classifier``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = config["data"]
    model_cfg = config["model"]

    features, labels = generate_dataset(
        data_cfg["n_instances"],
        data_cfg["n_timepoints"],
        data_cfg["seed"],
    )
    plot_sample_series(
        features,
        list(SAMPLE_CLASS_LABELS),
        output_dir / "sample_series.png",
    )
    x_train, x_test, y_train, y_test = split_data(
        features,
        labels,
        test_size=model_cfg["test_size"],
        random_state=model_cfg["random_state"],
    )
    classifier = fit_classifier(x_train, y_train, model_cfg["n_neighbors"])
    results = evaluate_classifier(classifier, x_test, y_test)
    return features, labels, results


def log_summary(
    features: np.ndarray,
    labels: np.ndarray,
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Log run metrics after the pipeline completes."""
    logger.info("Accuracy: %.2f%%", results["accuracy"] * 100)
    logger.info("Data shape: %s", features.shape)
    logger.info("Number of classes: %s", len(np.unique(labels)))
    logger.info("Class distribution: %s", np.bincount(labels))
    logger.info("Analysis complete. Figures saved to %s", output_dir)


def main(argv: list[str] | None = None) -> None:
    """Run the aeon clustering demo end to end."""
    configure_logging()
    args = parse_args(argv)
    config = load_config(args.config)
    output_dir = resolve_output_dir(args, config)

    features, labels, results = run_pipeline(config, output_dir)
    log_summary(features, labels, results, output_dir)


if __name__ == "__main__":
    main()
