"""
Time Series Clustering with AEON
Using specialized time series clustering algorithms from the AEON library
"""

import signalplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# AEON imports for time series clustering
from aeon.clustering import TimeSeriesKMeans
from aeon.distances import dtw_distance

np.random.seed(42)
signalplot.apply(font_family='serif')




@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    n_clusters: int = 3
    distance: str = "dtw"  # Dynamic Time Warping distance


def load_series(cfg: Config) -> pd.Series:
    """Load EIA electricity generation time series"""
    p = Path(cfg.csv_path)
    if not p.exists():
        raise FileNotFoundError(f"{cfg.csv_path} not found")

    df = pd.read_csv(p, header=None, usecols=[0, 1], names=["date", "value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def prepare_annual_sequences(s: pd.Series) -> tuple[np.ndarray, list]:
    """Prepare annual sequences for time series clustering"""
    df = s.to_frame("value")
    df["year"] = df.index.year
    df["month"] = df.index.month

    # Get complete years only (12 months)
    year_counts = df.groupby("year").size()
    complete_years = year_counts[year_counts == 12].index

    df_complete = df[df["year"].isin(complete_years)]

    # Create 3D array: (n_samples, n_channels=1, n_timepoints=12)
    sequences = []
    years = []

    for year in complete_years:
        year_data = (
            df_complete[df_complete["year"] == year]
            .sort_values("month")["value"]
            .values
        )
        if len(year_data) == 12:
            sequences.append(year_data)
            years.append(year)

    # AEON expects shape (n_samples, n_channels, n_timepoints)
    X = np.array(sequences).reshape(len(sequences), 1, 12)

    return X, years


def compute_dtw_matrix(sequences: np.ndarray) -> np.ndarray:
    """Compute pairwise DTW distance matrix"""
    n = sequences.shape[0]
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = dtw_distance(sequences[i, 0], sequences[j, 0])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix


def main(plot: bool = False):
    cfg = Config()

    try:
        s = load_series(cfg)
    except FileNotFoundError:
        logger.info(f"NOTE: {cfg.csv_path} not found, cannot run AEON clustering")
        return

    logger.info(f"\nTime Series Clustering with AEON")
    logger.info("=== Total observations: {len(s)} ===")
    logger.info(f"Date range: {s.index.min().date()} to {s.index.max().date()}")

    # Prepare annual sequences
    X, years = prepare_annual_sequences(s)
    logger.info(f"Complete years: {len(years)}")
    logger.info(f"Shape for clustering: {X.shape} (samples, channels, timepoints)")

    # AEON TimeSeriesKMeans with DTW distance
    logger.info(f"\nRunning TimeSeriesKMeans (distance={cfg.distance})...")
    clusterer = TimeSeriesKMeans(
        n_clusters=cfg.n_clusters,
        distance=cfg.distance,
        n_init=10,
        random_state=42,
        max_iter=50,
    )

    labels = clusterer.fit_predict(X)

    # Cluster statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    logger.info(f"\nCluster Distribution:")
    for label, count in zip(unique_labels, counts):
        cluster_years = [years[i] for i in range(len(years)) if labels[i] == label]
        logger.info(f"  Cluster {label}: {count} years ({count/len(years)*100:.1f}%)")
        logger.info(f"    Years: {cluster_years}")

    # Compute DTW distance matrix for visualization
    logger.info(f"\nComputing DTW distance matrix...")
    dtw_matrix = compute_dtw_matrix(X)

    # Create visualizations
    if plot:
        fig = plt.figure(figsize=(16, 10))

    # 1. Cluster assignments over time
        ax1 = plt.subplot(2, 3, 1)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for i, year in enumerate(years):
            ax1.scatter(year, labels[i], c=colors[labels[i]], s=100, alpha=0.7)
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Cluster")
        ax1.set_title("Cluster Assignments Over Time")
        ax1.set_yticks(range(cfg.n_clusters))

    # 2. Cluster centroids (seasonal patterns)
        ax2 = plt.subplot(2, 3, 2)
        months = range(1, 13)
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        for k in range(cfg.n_clusters):
            centroid = clusterer.cluster_centers_[k, 0, :]  # Shape: (n_timepoints,)
            ax2.plot(
                months,
                centroid,
                label=f"Cluster {k}",
                linewidth=2.5,
                marker="o",
                color=colors[k],
            )

        ax2.set_xlabel("Month")
        ax2.set_ylabel("Generation (thousand MWh)")
        ax2.set_title("Cluster Centroids (Seasonal Patterns)")
        ax2.set_xticks(months)
        ax2.set_xticklabels(month_names, rotation=45, ha="right")
        ax2.legend()

    # 3. All sequences colored by cluster
        ax3 = plt.subplot(2, 3, 3)
        for i in range(len(X)):
            ax3.plot(months, X[i, 0, :], color=colors[labels[i]], alpha=0.3, linewidth=1)

        ax3.set_xlabel("Month")
        ax3.set_ylabel("Generation (thousand MWh)")
        ax3.set_title("All Annual Patterns by Cluster")
        ax3.set_xticks(months)
        ax3.set_xticklabels(month_names, rotation=45, ha="right")

    # 4. DTW distance heatmap
        ax4 = plt.subplot(2, 3, 4)
        im = ax4.imshow(dtw_matrix, cmap="YlOrRd", aspect="auto")
        ax4.set_xlabel("Year Index")
        ax4.set_ylabel("Year Index")
        ax4.set_title("DTW Distance Matrix")
        plt.colorbar(im, ax=ax4, label="DTW Distance")

    # 5. Sample sequences from each cluster
        ax5 = plt.subplot(2, 3, 5)
        for k in range(cfg.n_clusters):
            cluster_indices = np.where(labels == k)[0]
            if len(cluster_indices) > 0:
                sample_idx = cluster_indices[0]
                ax5.plot(
                    months,
                    X[sample_idx, 0, :],
                    label=f"Cluster {k} ({years[sample_idx]})",
                    linewidth=2,
                    marker="o",
                    color=colors[k],
                )

        ax5.set_xlabel("Month")
        ax5.set_ylabel("Generation (thousand MWh)")
        ax5.set_title("Representative Year from Each Cluster")
        ax5.set_xticks(months)
        ax5.set_xticklabels(month_names, rotation=45, ha="right")
        ax5.legend()

    # 6. Inertia and silhouette score
        ax6 = plt.subplot(2, 3, 6)
        from sklearn.metrics import silhouette_score

    # Flatten for silhouette score
        X_flat = X.reshape(X.shape[0], -1)
        sil_score = silhouette_score(X_flat, labels, metric="euclidean")

        metrics_text = (
            f"Clustering Metrics:\n\n"
            f"Distance: {cfg.distance.upper()}\n"
            f"Clusters: {cfg.n_clusters}\n"
            f"Inertia: {clusterer.inertia_:.1f}\n"
            f"Silhouette: {sil_score:.3f}\n"
            f"Iterations: {clusterer.n_iter_}\n\n"
            f"Interpretation:\n"
            f"- DTW captures shape similarity\n"
            f"- Clusters reveal distinct\n"
            f"  seasonal patterns\n"
            f"- Higher silhouette = better\n"
            f"  separation"
        )

        ax6.text(
            0.1,
            0.5,
            metrics_text,
            transform=ax6.transAxes,
            fontsize=11,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )
        ax6.axis("off")

        signalplot.save("eia_aeon_ts_clusters.png")

    logger.info(f"\nClustering Quality:")
    logger.info(f"  Inertia:         {clusterer.inertia_:.1f}")
    logger.info(f"  Silhouette:      {sil_score:.3f}")
    logger.info(f"  Iterations:      {clusterer.n_iter_}")

    logger.info(f"\nOutput: eia_aeon_ts_clusters.png\n")


if __name__ == "__main__":
    main()
