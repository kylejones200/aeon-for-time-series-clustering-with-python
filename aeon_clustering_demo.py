"""K-neighbors time series classification demo with aeon."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.datasets import make_example_3_class_dataset
from aeon.visualisation import plot_series
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    X, y = make_example_3_class_dataset(n_instances=50, n_timepoints=30, random_state=42)

    plot_series(X[0], X[1], X[2], labels=["Sine", "Cosine", "Sine2x"])
    plt.title("Sample Series from Each Class")
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    clf = KNeighborsTimeSeriesClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    logging.info("Accuracy: %.2f", accuracy_score(y_test, y_pred))
    logging.info("Data shape: %s", X.shape)
    logging.info("Number of classes: %d", len(set(y)))
    logging.info("Class distribution: %s", np.bincount(y))


if __name__ == "__main__":
    main()
