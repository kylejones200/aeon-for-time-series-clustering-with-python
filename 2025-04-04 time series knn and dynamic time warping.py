"""Generated from Jupyter notebook: 2025-04-04 time series knn and dynamic time warping

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

import numpy as np

# Create a synthetic univariate time series dataset
X = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
print(X.shape)  # (3 samples, 5 timesteps)
"""
Resampling Time Series
Adjust time series to have a consistent length.
"""
from tslearn.preprocessing import TimeSeriesResampler

# Resample to 10 timesteps
resampler = TimeSeriesResampler(sz=10)
X_resampled = resampler.fit_transform(X)
print(X_resampled.shape)  # (3 samples, 10 timesteps)
"""
Normalizing Time Series
Standardize time series to have zero mean and unit variance.
"""
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

scaler = TimeSeriesScalerMeanVariance()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
"""
Clustering Time Series
Now to the fun stuff. tslearn provides specialized algorithms for clustering, such as K-Shape and DTW-based K-Means.
K-Shape Clustering
K-Shape clusters time series based on shape similarity.
"""
import matplotlib.pyplot as plt
from tslearn.clustering import KShape

# Synthetic time series dataset
X = np.random.rand(100, 50, 1)  # 100 samples, 50 timesteps

# Apply K-Shape clustering
kshape = KShape(n_clusters=3, random_state=0)
y_pred = kshape.fit_predict(X)

# Plot cluster centroids
for centroid in kshape.cluster_centers_:
    plt.plot(centroid.ravel())
plt.title("K-Shape Cluster Centroids")
plt.show()
"""
DTW K-Means
Use Dynamic Time Warping (DTW) for clustering.
"""
from tslearn.clustering import TimeSeriesKMeans

# Apply DTW-based K-Means
dtw_kmeans = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=0)
y_pred = dtw_kmeans.fit_predict(X)

# Plot cluster centroids
for centroid in dtw_kmeans.cluster_centers_:
    plt.plot(centroid.ravel())
plt.title("DTW K-Means Cluster Centroids")
plt.show()
"""
Time Series Classification
tslearn supports time series classification using k-Nearest Neighbors (kNN) and other methods.
kNN Classification with DTW
"""
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

# Create synthetic dataset
X = np.random.rand(200, 50, 1)  # 200 samples, 50 timesteps
y = np.random.randint(0, 2, 200)  # Binary labels

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a k-NN classifier with DTW
knn = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric="dtw")
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
"""
Dynamic Time Warping (DTW)
DTW measures the similarity between two time series by aligning them.
So we can compute the distance between two series using DTW Distance
"""
from tslearn.metrics import dtw

# Two synthetic time series
ts1 = np.array([1, 2, 3, 4, 5])
ts2 = np.array([2, 3, 4, 5, 6])

# Compute DTW distance
distance = dtw(ts1, ts2)
print(f"DTW Distance: {distance:.2f}")
"""
Feature Extraction
Example: Symbolic Aggregate Approximation (SAX)
Convert time series into symbolic representations.
"""
from tslearn.piecewise import SymbolicAggregateApproximation

# Apply SAX
sax = SymbolicAggregateApproximation(n_segments=5, alphabet_size_avg=3)
X_sax = sax.fit_transform(X)
print(X_sax)


# --- code cell ---

# !pip install tslearn  # Jupyter-only


# --- code cell ---

import numpy as np
from classification import train_knn_classifier
from clustering import dtw_kmeans_clustering, kshape_clustering
from dtw_analysis import compute_dtw_distance
from feature_extraction import sax_transformation
from preprocessing import normalize_time_series, resample_time_series


def main():
    print("Running Time Series Analysis...\n")

    # Synthetic data
    X = np.random.rand(100, 50, 1)
    y = np.random.randint(0, 2, 100)

    # Resampling & Normalization
    X_resampled = resample_time_series(X, new_size=10)
    X_scaled = normalize_time_series(X)

    # Clustering
    kshape_model, _ = kshape_clustering(X)
    dtw_kmeans_model, _ = dtw_kmeans_clustering(X)

    # Classification
    accuracy = train_knn_classifier(X, y)
    print(f"K-NN Classifier Accuracy: {accuracy:.2f}")

    # DTW Analysis
    ts1 = np.array([1, 2, 3, 4, 5])
    ts2 = np.array([2, 3, 4, 5, 6])
    distance = compute_dtw_distance(ts1, ts2)
    print(f"DTW Distance: {distance:.2f}")

    # Feature Extraction
    X_sax = sax_transformation(X)
    print("SAX Feature Extraction Complete.")


if __name__ == "__main__":
    main()


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
from classification import train_knn_classifier
from clustering import dtw_kmeans_clustering, kshape_clustering
from dtw_analysis import compute_dtw_distance
from feature_extraction import sax_transformation
from preprocessing import normalize_time_series, resample_time_series
from visualization import plot_clusters, plot_time_series


def main():
    print("Running Time Series Analysis...\n")

    # Generate synthetic time series data
    X = np.random.rand(100, 50, 1)
    y = np.random.randint(0, 2, 100)

    # Resampling & Normalization
    X_resampled = resample_time_series(X, new_size=10)
    X_scaled = normalize_time_series(X)

    # Plot sample time series
    plot_time_series(X, title="Original Time Series Data")
    plot_time_series(X_scaled, title="Normalized Time Series Data")

    # Clustering
    kshape_model, _ = kshape_clustering(X)
    dtw_kmeans_model, _ = dtw_kmeans_clustering(X)

    # Plot clustering results
    plot_clusters(kshape_model, "K-Shape Cluster Centroids")
    plot_clusters(dtw_kmeans_model, "DTW K-Means Cluster Centroids")

    # Classification
    accuracy = train_knn_classifier(X, y)
    print(f"K-NN Classifier Accuracy: {accuracy:.2f}")

    # DTW Analysis
    ts1 = np.array([1, 2, 3, 4, 5])
    ts2 = np.array([2, 3, 4, 5, 6])
    distance = compute_dtw_distance(ts1, ts2)
    print(f"DTW Distance: {distance:.2f}")

    # Feature Extraction
    X_sax = sax_transformation(X)
    print("SAX Feature Extraction Complete.")


if __name__ == "__main__":
    main()
