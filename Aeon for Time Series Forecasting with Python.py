#!/usr/bin/env python3
"""
Aeon for Time Series Forecasting with Python.

Tutorial-style demos: visualization, classification, regression, clustering,
transformations, segmentation, and forecasting. Converted from the companion
notebook; run with: python "Aeon for Time Series Forecasting with Python.py"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.clustering import TimeSeriesKMeans
from aeon.datasets import (
    load_airline,
    load_electric_devices_segmentation,
    load_gun_point_segmentation,
    load_longley,
    make_example_3_class_dataset,
)
from aeon.forecasting import ETSForecaster
from aeon.performance_metrics.forecasting import mean_absolute_error
from aeon.regression.interval_based import TimeSeriesForestRegressor
from aeon.segmentation import ClaSPSegmenter, find_dominant_window_sizes
from aeon.transformations.series.scaling import Scaler
from aeon.visualisation import (
    plot_series,
    plot_series_with_change_points,
    plot_series_with_profiles,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def make_sine_cosine_dataset(
    n_samples: int = 50, n_timepoints: int = 30, random_state: int = 42
) -> np.ndarray:
    """Synthetic univariate series (sin, cos, sin(2t)) with noise."""
    rng = np.random.default_rng(random_state)
    t = np.linspace(0, 2 * np.pi, n_timepoints)
    patterns = [
        lambda: np.sin(t),
        lambda: np.cos(t),
        lambda: 0.5 * np.sin(2 * t),
    ]
    X = []
    for _ in range(n_samples):
        series = patterns[rng.integers(0, len(patterns))]() + rng.normal(0, 0.1, n_timepoints)
        X.append(series)
    return np.array(X)


def prepare_longley_regression() -> tuple[np.ndarray, np.ndarray]:
    """Reshape Longley data for aeon regression (3D X, 1D y)."""
    X, y = load_longley()
    X = pd.DataFrame(X).values.reshape((len(X), -1, 1))
    y = y.iloc[:, 0].values
    return X, y


def plot_cluster_subplots(X: np.ndarray, labels: np.ndarray, n_clusters: int = 3) -> None:
    """Plot each cluster's series in a stacked subplot."""
    fig, axes = plt.subplots(n_clusters, 1, figsize=(15, 10), sharex=True)
    if n_clusters == 1:
        axes = [axes]
    for cluster, ax in enumerate(axes):
        for series in X[labels == cluster]:
            ax.plot(series, alpha=0.5)
        ax.set_title(f"Cluster {cluster + 1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
    fig.tight_layout()


def demo_visualize_airline() -> None:
    """Basic line plot of airline passenger numbers."""
    y = load_airline()
    plot_series(y, title="Airline Passenger Numbers")
    plt.show()


def demo_visualize_longley() -> None:
    """Plot multivariate Longley series with aeon."""
    X, _y = load_longley()
    plot_series(X)
    plt.show()


def demo_visualize_longley_matplotlib() -> None:
    """Matplotlib plot of Longley target columns over time."""
    _X, y = load_longley()
    y = y.copy()
    y.index = y.index.to_timestamp()
    plt.figure(figsize=(12, 6))
    for column in y.columns:
        plt.plot(y.index, y[column], label=column)
    plt.title("Longley Dataset")
    plt.xlabel("Year")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def demo_classify_synthetic() -> None:
    """Time series forest classifier on a 3-class synthetic dataset."""
    X, y = make_example_3_class_dataset(n_instances=100, n_timepoints=50, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = TimeSeriesForestClassifier()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    logger.info("Classification accuracy: %.2f", accuracy)


def demo_regression_longley() -> None:
    """Time series forest regressor on reshaped Longley data."""
    X, y = prepare_longley_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = TimeSeriesForestRegressor(random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    logger.info("R-squared score: %.2f", r2)


def demo_regression_longley_plots() -> None:
    """Regression on Longley with diagnostic plots."""
    X, y = prepare_longley_regression()
    plt.figure(figsize=(12, 6))
    for i in range(X.shape[0]):
        plt.plot(X[i, :, 0], label=f"Series {i + 1}" if i < 5 else "")
    plt.title("Original Time Series Data")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = TimeSeriesForestRegressor(random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    logger.info("R-squared score: %.2f", r2_score(y_test, y_pred))
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color="blue", alpha=0.5)
    lo, hi = y_test.min(), y_test.max()
    plt.plot([lo, hi], [lo, hi], "r--", lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual Values")
    plt.show()
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    if HAS_SEABORN:
        sns.histplot(residuals, kde=True)
    else:
        plt.hist(residuals, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, color="green", alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values")
    plt.show()


def demo_cluster_synthetic() -> tuple[np.ndarray, np.ndarray]:
    """K-means clustering on sine/cosine synthetic data; returns X and labels."""
    X = make_sine_cosine_dataset(n_samples=50, n_timepoints=30, random_state=42)
    kmeans = TimeSeriesKMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    plot_cluster_subplots(X, labels)
    plt.show()
    logger.info("Shape of X: %s", X.shape)
    logger.info("Shape of labels: %s", labels.shape)
    return X, labels


def demo_classify_cluster_labels(X: np.ndarray, labels: np.ndarray) -> None:
    """Train a classifier using cluster IDs as pseudo-labels."""
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    clf = TimeSeriesForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    logger.info("Accuracy: %.2f", accuracy)
    logger.info("Data shape: %s", X.shape)
    logger.info("Number of classes: %d", len(np.unique(labels)))
    logger.info("Class distribution: %s", np.bincount(labels))


def demo_cluster_noisy_random_walk() -> tuple[np.ndarray, np.ndarray]:
    """K-means on random-walk series; returns X and labels."""
    rng = np.random.default_rng(42)
    n_series, n_timepoints = 30, 100
    X = np.array([np.cumsum(rng.standard_normal(n_timepoints)) for _ in range(n_series)])
    kmeans = TimeSeriesKMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    plot_cluster_subplots(X, labels)
    plt.show()
    return X, labels


def demo_scale_airline() -> None:
    """Scale airline series and compare original vs scaled."""
    y = load_airline()
    scaler = Scaler()
    y_scaled = scaler.fit_transform(y)
    plot_series(y, y_scaled, labels=["Original", "Scaled"])
    plt.show()


def demo_segmentation_gunpoint() -> pd.DataFrame:
    """Load gun-point segmentation change points."""
    _X, _period_length, change_points = load_gun_point_segmentation()
    df = pd.DataFrame(change_points)
    logger.info("Gun-point change points:\n%s", df.head())
    return df


def demo_segmentation_electric_devices() -> None:
    """Plot electric-devices segmentation with change points."""
    ed_seg, _, ed_seg_chp = load_electric_devices_segmentation()
    plot_series_with_change_points(ed_seg, ed_seg_chp)
    plt.show()


def demo_segmentation_clasp(csv_path: Path) -> None:
    """ClaSP segmentation on a CSV column (expects an 'hd' column)."""
    if not csv_path.is_file():
        logger.warning("Skipping ClaSP demo: %s not found", csv_path)
        return

    df = pd.read_csv(csv_path)
    if "hd" not in df.columns:
        logger.warning("Skipping ClaSP demo: column 'hd' not in %s", csv_path)
        return

    clasp = ClaSPSegmenter(period_length=3, n_cps=3)
    found_cps = clasp.fit_predict(df["hd"].values)
    profiles = clasp.profiles
    logger.info("Found change points: %s", found_cps)
    dominant_period_size = find_dominant_window_sizes(df["hd"])
    logger.info("Dominant period size: %s", dominant_period_size)
    logger.info("Series length: %d", len(df))
    plot_series_with_profiles(
        df["hd"].values,
        profiles,
        found_cps=found_cps,
        title="Electric Devices",
    )
    plt.show()


def demo_forecast_ets_simple() -> None:
    """One-step ETS forecast on airline data."""
    y = load_airline()
    forecaster = ETSForecaster(alpha=0.4, beta=0.2, gamma=0.5, phi=0.8, horizon=1)
    forecaster.fit(y)
    prediction = forecaster.predict()
    logger.info("One-step forecast: %s", prediction)


def demo_forecast_ets_holdout() -> None:
    """12-month holdout forecast with MAE on airline passengers."""
    y = load_airline()
    train = y[:-12]
    test = y[-12:]
    forecaster = ETSForecaster(seasonal="add", sp=12)
    forecaster.fit(train)
    fh = np.arange(1, 13)
    y_pred = forecaster.predict(fh)
    plt.figure(figsize=(12, 6))
    train.plot(label="Training Data", color="blue")
    test.plot(label="Test Data", color="green")
    y_pred.plot(label="Forecast", color="red")
    plt.title("Airline Passengers Forecast using ETSForecaster")
    plt.xlabel("Year")
    plt.ylabel("Passengers")
    plt.legend()
    plt.grid(False)
    plt.show()
    logger.info("Forecast values:\n%s", y_pred)
    mae = mean_absolute_error(test, y_pred)
    logger.info("Mean Absolute Error: %.2f", mae)


DEMOS: dict[str, callable] = {
    "airline-plot": demo_visualize_airline,
    "longley-plot": demo_visualize_longley,
    "longley-mpl": demo_visualize_longley_matplotlib,
    "classify": demo_classify_synthetic,
    "regression": demo_regression_longley,
    "regression-plots": demo_regression_longley_plots,
    "scale": demo_scale_airline,
    "segment-gunpoint": demo_segmentation_gunpoint,
    "segment-electric": demo_segmentation_electric_devices,
    "forecast-simple": demo_forecast_ets_simple,
    "forecast-holdout": demo_forecast_ets_holdout,
}


def run_clustering_pipeline() -> None:
    """Cluster synthetic data, then classify using cluster labels."""
    X, labels = demo_cluster_synthetic()
    demo_classify_cluster_labels(X, labels)


def run_noisy_clustering_pipeline() -> None:
    """Cluster random-walk data, then classify using cluster labels."""
    X, labels = demo_cluster_noisy_random_walk()
    demo_classify_cluster_labels(X, labels)


def run_all(csv_path: Path) -> None:
    """Run every demo in notebook order."""
    demo_visualize_airline()
    demo_visualize_longley()
    demo_classify_synthetic()
    demo_regression_longley()
    run_clustering_pipeline()
    demo_scale_airline()
    demo_visualize_longley_matplotlib()
    demo_regression_longley_plots()
    run_noisy_clustering_pipeline()
    demo_segmentation_gunpoint()
    demo_segmentation_electric_devices()
    demo_segmentation_clasp(csv_path)
    demo_forecast_ets_simple()
    demo_forecast_ets_holdout()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aeon time series forecasting and analysis demos")
    parser.add_argument(
        "--demo",
        choices=[*DEMOS.keys(), "cluster", "noisy-cluster", "clasp", "all"],
        default="all",
        help="Which demo to run (default: all)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/hd.csv"),
        help="CSV path for ClaSP segmentation (column 'hd')",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    if args.demo == "all":
        run_all(args.csv)
    elif args.demo == "cluster":
        run_clustering_pipeline()
    elif args.demo == "noisy-cluster":
        run_noisy_clustering_pipeline()
    elif args.demo == "clasp":
        demo_segmentation_clasp(args.csv)
    else:
        DEMOS[args.demo]()


if __name__ == "__main__":
    main()
