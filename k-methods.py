import argparse
from pathlib import Path

import numpy as np
import kmedoids
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def k_medoids(X, n_clusters=3, *, distances=None):
    X = np.asarray(X, dtype=float)
    if distances is None:
        distances = squareform(pdist(X, metric="euclidean"))
    model = kmedoids.fasterpam(distances, medoids=n_clusters, random_state=42)
    return model


class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, eps=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.eps = eps
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        indices = rng.choice(len(X), self.n_clusters, replace=False)
        self.cluster_centers = X[indices].copy()

        for iter in range(self.max_iter):
            distances = np.linalg.norm(X[:, None, :] - self.cluster_centers, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centers = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else self.cluster_centers[i]
                                    for i in range(self.n_clusters)])

            shift = np.linalg.norm(new_centers - self.cluster_centers)
            self.cluster_centers = new_centers

            if shift < self.eps:
                break

        self.labels = self.predict(X)
        self.inertia = np.sum((X - self.cluster_centers[self.labels]) ** 2)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = np.linalg.norm(X[:, None, :] - self.cluster_centers, axis=2)
        return np.argmin(distances, axis=1)


def load_data(path="customers.csv"):
    df = pd.read_csv(path)
    features = df.drop(columns=["Row", "CustomerId", "Defaulted"], errors="ignore")
    if features.empty or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("CSV must contain rows and finite numeric clustering features")
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    return X, features

def plot_clusters(features, labels, title, x_col="Income", y_col="DebtIncomeRatio", *, output=None):
    plt.figure(figsize=(9, 6))
    plt.scatter(features[x_col], features[y_col], c=labels, cmap="tab10", s=25, alpha=0.7)
    plt.xlabel(f"{x_col} (standardized)")
    plt.ylabel(f"{y_col} (standardized)")
    plt.title(title)
    plt.colorbar(label="Cluster")
    plt.grid(True)
    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output, dpi=140)
        plt.close()

def plot_errors(k_values, kmeans_errors, kmedoids_errors, *, output=None, noise_percent=0):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fig.suptitle(f"Noise: {noise_percent:g}%")
    axes[0].plot(k_values, kmeans_errors, marker="o")
    axes[0].set_title("k-Means")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia (squared distances)")

    axes[1].plot(k_values, kmedoids_errors, marker="o", color="orange")
    axes[1].set_title("k-Medoids")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Loss (distances)")

    for ax in axes:
        ax.set_xticks(k_values)
        ax.grid(True)

    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output, dpi=140)
        plt.close()

def add_noise(X, noise_ratio=0.0, noise_strength=3.0, random_state=42):
    rng = np.random.default_rng(random_state)
    X_noisy = X.copy()

    n_noisy = round(len(X) * noise_ratio)

    if n_noisy == 0:
        return X_noisy

    indices = rng.choice(len(X), size=n_noisy, replace=False)

    # Для каждого выбранного объекта изменяем одну случайную координату
    columns = rng.integers(0, X.shape[1], size=n_noisy)

    noise = rng.normal(0, noise_strength, size=n_noisy)

    X_noisy[indices, columns] += noise

    return X_noisy

def main():
    parser = argparse.ArgumentParser(description="Cluster numeric CSV data using k-Means and k-Medoids")
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("customers.csv"))
    parser.add_argument("--k", type=int, nargs="+", default=list(range(3, 10)))
    parser.add_argument("--noise", type=float, nargs="+", default=[0, 1, 3, 5, 10],
                        metavar="PERCENT", help="Share of perturbed rows, from 0 to 100")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if any(not 0 <= percent <= 100 for percent in args.noise):
        parser.error("--noise percentages must be between 0 and 100")

    try:
        X, features = load_data(args.csv)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if any(not 1 <= k <= len(X) for k in args.k):
        parser.error(f"--k must be between 1 and {len(X)}")
    reserved = {"kmeans_cluster", "kmedoids_cluster"}
    if reserved.intersection(features.columns):
        parser.error("CSV feature names conflict with output cluster columns")

    k_values = sorted(set(args.k))
    summary = []
    for percent in dict.fromkeys(args.noise):
        percent = 0.0 if percent == 0 else percent
        folder = args.output_dir / f"noise_{percent}"
        folder.mkdir(parents=True, exist_ok=True)
        X_noisy = add_noise(X, noise_ratio=percent / 100)
        distances = squareform(pdist(X_noisy, metric="euclidean"))
        coordinates = pd.DataFrame(X_noisy, columns=features.columns)
        x_col = "Income" if "Income" in features else features.columns[0]
        y_col = "DebtIncomeRatio" if "DebtIncomeRatio" in features else next(
            (column for column in features if column != x_col), None
        )
        if y_col is None:
            y_col = "Projection baseline"
            while y_col in coordinates:
                y_col += "_"
            plot_coordinates = coordinates.assign(**{y_col: 0})
        else:
            plot_coordinates = coordinates

        kmeans_errors = []
        kmedoids_errors = []
        for k in k_values:
            km = KMeans(n_clusters=k).fit(X_noisy)
            kmed = k_medoids(X_noisy, n_clusters=k, distances=distances)
            kmeans_errors.append(km.inertia)
            kmedoids_errors.append(kmed.loss)
            summary.append({
                "noise_percent": percent,
                "perturbed_rows": round(len(X) * percent / 100),
                "k": k,
                "kmeans_inertia": km.inertia,
                "kmedoids_loss": kmed.loss,
            })
            print(f"Noise={percent:g}%, k={k}: k-Means={km.inertia:.2f}, k-Medoids={kmed.loss:.2f}")
            coordinates.assign(kmeans_cluster=km.labels, kmedoids_cluster=kmed.labels).to_csv(
                folder / f"k_{k}_assignments.csv", index=False
            )
            for method, labels in (("k-Means", km.labels), ("k-Medoids", kmed.labels)):
                plot_clusters(
                    plot_coordinates, labels, f"{method} | k={k} | Noise={percent:g}%",
                    x_col, y_col, output=folder / f"k_{k}_{method.lower()}.png"
                )

        plot_errors(k_values, kmeans_errors, kmedoids_errors,
                    output=folder / "errors.png", noise_percent=percent)
    pd.DataFrame(summary).to_csv(args.output_dir / "errors.csv", index=False)


if __name__ == "__main__":
    main()
