"""Time series k-NN and dynamic time warping with tslearn.

Demonstrates preprocessing, clustering, classification, DTW distance,
and SAX feature extraction on synthetic data.
"""

from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tslearn.clustering import KShape, TimeSeriesKMeans
from tslearn.metrics import dtw
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.preprocessing import TimeSeriesResampler, TimeSeriesScalerMeanVariance

logger = logging.getLogger(__name__)


def demo_preprocessing() -> None:
    """Resample and normalize a small univariate dataset."""
    x = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
    logger.info("Original shape: %s", x.shape)
    resampler = TimeSeriesResampler(sz=10)
    x_resampled = resampler.fit_transform(x)
    logger.info("Resampled shape: %s", x_resampled.shape)
    scaler = TimeSeriesScalerMeanVariance()
    x_scaled = scaler.fit_transform(x)
    logger.info("Scaled series (first sample):\n%s", x_scaled[0].ravel())


def plot_cluster_centroids(model, title: str) -> None:
    for centroid in model.cluster_centers_:
        plt.plot(centroid.ravel())
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()


def demo_clustering(
    x: np.ndarray,
    *,
    n_clusters: int = 3,
    random_state: int = 0,
    show_plots: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster time series with K-Shape and DTW K-Means."""
    kshape = KShape(n_clusters=n_clusters, random_state=random_state)
    kshape_labels = kshape.fit_predict(x)
    logger.info("K-Shape cluster sizes: %s", np.bincount(kshape_labels))
    if show_plots:
        plot_cluster_centroids(kshape, "K-Shape Cluster Centroids")
        plt.show()

    dtw_kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=random_state)
    dtw_labels = dtw_kmeans.fit_predict(x)
    logger.info("DTW K-Means cluster sizes: %s", np.bincount(dtw_labels))
    if show_plots:
        plot_cluster_centroids(dtw_kmeans, "DTW K-Means Cluster Centroids")
        plt.show()

    return kshape_labels, dtw_labels


def demo_classification(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_neighbors: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
) -> float:
    """Train a k-NN classifier with DTW and return test accuracy."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=n_neighbors, metric="dtw")
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info("K-NN (DTW) accuracy: %.2f", accuracy)
    return accuracy


def demo_dtw_distance() -> float:
    """Compute DTW distance between two short series."""
    ts1 = np.array([1, 2, 3, 4, 5])
    ts2 = np.array([2, 3, 4, 5, 6])
    distance = dtw(ts1, ts2)
    logger.info("DTW distance: %.2f", distance)
    return distance


def demo_sax(x: np.ndarray, *, n_segments: int = 5, alphabet_size: int = 3) -> np.ndarray:
    """Convert time series to symbolic aggregate approximations."""
    sax = SymbolicAggregateApproximation(n_segments=n_segments, alphabet_size_avg=alphabet_size)
    x_sax = sax.fit_transform(x)
    logger.info("SAX representation (first sample): %s", x_sax[0].ravel())
    return x_sax


def make_clustering_data(
    n_samples: int = 100,
    n_timesteps: int = 50,
    *,
    random_state: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    return rng.random((n_samples, n_timesteps, 1))


def make_classification_data(
    n_samples: int = 200,
    n_timesteps: int = 50,
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    x = rng.random((n_samples, n_timesteps, 1))
    y = rng.integers(0, 2, n_samples)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Time series k-NN and DTW examples with tslearn")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip interactive cluster plots",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=== Preprocessing ===")
    demo_preprocessing()
    logger.info("\n=== Clustering ===")
    x_cluster = make_clustering_data(random_state=args.seed)
    demo_clustering(x_cluster, random_state=args.seed, show_plots=not args.no_plots)
    logger.info("\n=== Classification ===")
    x_class, y_class = make_classification_data(random_state=args.seed + 1)
    demo_classification(x_class, y_class, random_state=args.seed + 1)
    logger.info("\n=== DTW distance ===")
    demo_dtw_distance()
    logger.info("\n=== SAX feature extraction ===")
    demo_sax(x_class)


if __name__ == "__main__":
    main()
