import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans,
    SpectralClustering,
    DBSCAN,
    AgglomerativeClustering,
)
from sklearn.mixture import GaussianMixture
import hdbscan
from kDBCV import DBCV_score
from sklearn.metrics import silhouette_score
from collections import Counter
from itertools import product
from typing import Callable


def fit_predict_clusters(embeddings, clusterer):
    X = normalize(np.array(embeddings))
    labels = clusterer.fit_predict(X)
    return X, labels


def optimize_clusterer(
    embeddings: np.ndarray,
    model: KMeans | hdbscan.HDBSCAN,
    metric: Callable,  # DBCV_score | silhouette_score
    exploration_parameters: list[dict],
):
    results = []
    best_score = float("-inf")
    best_parameters = {}

    param_keys = list(exploration_parameters.keys())
    param_values = list(exploration_parameters.values())
    # try all combinations of hyperparam values
    for p in product(*param_values):
        parameters = dict(zip(param_keys, p))
        model.set_params(**parameters)

        X, labels = fit_predict_clusters(embeddings, model)

        score = metric(X, labels)

        if score > best_score:
            best_score = score
            best_parameters = parameters

        results.append({"parameters": parameters, "score": score})

    return results, best_parameters


def graph_parameter_search_results(ax, results: list[dict]):
    """
    Dynamic plot based on dim of results (num of parameters)
    1d = line plot, parameter vs score
    2d = scatter plot, parameter vs parameter w/ heatmap color
    3d+ = line plot, index vs score (fallback option)
    results should be non-empty and have at least 1 parameter
    """
    assert len(results) > 0 and len(results[0]) > 0

    param_keys = list(results[0]["parameters"].keys())
    if len(param_keys) == 1:
        # line plot
        key = param_keys[0]
        x = [r["parameters"][key] for r in results]
        y = [r["score"] for r in results]

        ax.plot(x, y)
        ax.set_xlabel(key)
        ax.set_ylabel("Score")
    elif len(param_keys) == 2:
        # scatter plot
        key_x, key_y = param_keys
        x = [r["parameters"][key_x] for r in results]
        y = [r["parameters"][key_y] for r in results]
        scores = [r["score"] for r in results]

        scatter = ax.scatter(x, y, c=scores)

        ax.set_xlabel(key_x)
        ax.set_ylabel(key_y)
        plt.colorbar(scatter, ax=ax)
    else:
        # line plot fallback
        scores = [r["score"] for r in results]
        x = list(range(len(scores)))

        ax.plot(x, scores, marker="o")

        # annotate each point with param combination
        for i, r in enumerate(results):
            param_str = ", ".join(f"{k}={v}" for k, v in r["parameters"].items())
            ax.text(
                i,
                scores[i],
                param_str,
                fontsize=7,
                rotation=45,
                ha="right",
            )
        ax.set_xlabel("Experiment #")
        ax.set_ylabel("Score")
    ax.set_title(f"Hyperparameter Search Results")


def plot_PCA(ax, X, labels, title_suffix=""):
    noise_mask = labels == -1
    num_clusters = len(set(labels) - {-1})
    X_new = PCA(n_components=2).fit_transform(X)

    cmap = plt.get_cmap("tab20" if num_clusters <= 20 else "turbo")
    unique_labels = sorted(set(labels) - {-1})
    colors = {
        label: cmap(i / max(len(unique_labels) - 1, 1))
        for i, label in enumerate(unique_labels)
    }

    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(
            X_new[mask, 0],
            X_new[mask, 1],
            c=[colors[lbl]],
            s=12,
            alpha=0.9,
            linewidths=0,
            label=f"Cluster {lbl}",
        )

    if noise_mask.any():
        ax.scatter(
            X_new[noise_mask, 0],
            X_new[noise_mask, 1],
            c="lightgrey",
            s=8,
            alpha=0.5,
            linewidths=0,
            label="noise",
        )

    ax.set_title(f"PCA Projection {title_suffix}", fontsize=12)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()


def plot_cluster_sizes(ax, labels):
    counts = Counter(labels)  # dict of {item: count}, pretty cool
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    cluster_ids, sizes = zip(*sorted_counts)

    x = np.arange(len(sizes))
    # make noise stand out w/ different color
    bar_colors = ["#e74c3c" if c == -1 else "#3498db" for c in cluster_ids]

    bars = ax.bar(x, sizes, color=bar_colors, width=0.8)

    # annotate size
    for bar, size in zip(bars, sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(size),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    # label
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["noise" if c == -1 else str(c) for c in cluster_ids],
        rotation=90,
        fontsize=7,
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of points")
    ax.set_title("Cluster Sizes", fontsize=12)


def plot_cluster_similarity(ax, X, labels):
    unique_labels = sorted(set(labels))

    cluster_ids = []
    mean_similarities = []

    for label in unique_labels:
        mask = labels == label
        points = X[mask]

        centroid = np.mean(points, axis=0, keepdims=True)
        similarity = cosine_similarity(points, centroid).flatten()

        cluster_ids.append(label)
        mean_similarities.append(similarity.mean())

    # make noise stand out w/ different color
    bar_colors = ["#e74c3c" if c == -1 else "#3498db" for c in cluster_ids]

    x = np.arange(len(cluster_ids))
    bars = ax.bar(x, mean_similarities, color=bar_colors, width=0.8)

    # annotate size
    for bar, sim in zip(bars, mean_similarities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{sim:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    # label
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["noise" if c == -1 else str(c) for c in cluster_ids],
        rotation=90,
        fontsize=7,
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Avg Cosine Similarity")
    ax.set_title("Similarity to Cluster Center", fontsize=12)


def make_plots(
    embeddings: np.ndarray,
    clusterers: list[dict],  # [{"name", "model", "metric", "parameters"}]
    output_path="cluster_plots.png",
):
    # unique graph for each clusterer
    for clusterer in clusterers:
        fig, axes = plt.subplots(
            nrows=2, ncols=2, figsize=(12, 8), constrained_layout=True
        )
        # fig.tight_layout(rect=[0, 0, 0.85, 0.92])  # leave space for title and legend
        axes = axes.flatten()
        # find optimal hyperparameters for each clusterer using paired scoring function
        name = clusterer["name"]
        model = clusterer["model"]
        metric = clusterer["metric"]
        parameters = clusterer["parameters"]
        results, best_parameters = optimize_clusterer(
            embeddings,
            model,
            metric,
            parameters,
        )
        graph_parameter_search_results(axes[0], results)

        # display best parameters in top-right legend
        param_text = "\n".join(f"{k} = {v}" for k, v in best_parameters.items())
        fig.legend(
            [param_text],
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            frameon=True,
            title="Best Parameters",
            fontsize=8,
        )

        # get best performing clustering model
        model = model.set_params(**best_parameters)
        X, labels = fit_predict_clusters(embeddings, model)

        # explore best model performance
        plot_PCA(axes[1], X, labels)
        plot_cluster_sizes(axes[2], labels)
        plot_cluster_similarity(axes[3], X, labels)

        fig.suptitle(f"{name} Playlist Genre Clusters", fontsize=20)

        plt.savefig(f"{name}_{output_path}")
        plt.close()


if __name__ == "__main__":
    embeddings_path = (
        r"/home/ad.msoe.edu/whitcombp/MSOE/PlaylistGenreClassification/embeddings.json"
    )
    with open(embeddings_path, "r") as fp:
        data = json.load(fp)
    embeddings = data["embeddings"]

    clusterers = [
        {
            "name": "HDBSCAN",
            "model": hdbscan.HDBSCAN(),
            "metric": lambda x, y: DBCV_score(x, y)[0],
            "parameters": {
                "min_cluster_size": list(range(2, 25, 1)),
                "min_samples": list(range(1, 20, 1)),
            },
        },
        {
            "name": "KMeans",
            "model": KMeans(),
            "metric": silhouette_score,
            "parameters": {"n_clusters": list(range(2, 30, 2))},
        },
        {
            "name": "SpectralClustering",
            "model": SpectralClustering(assign_labels="kmeans"),
            "metric": silhouette_score,
            "parameters": {
                "n_clusters": list(range(2, 30, 2)),
                "affinity": ["nearest_neighbors", "rbf"],
            },
        },
        {
            "name": "DBSCAN",
            "model": DBSCAN(),
            "metric": lambda x, y: DBCV_score(x, y)[0],
            "parameters": {
                "eps": np.linspace(0.1, 1.5, 10).tolist(),
                "min_samples": list(range(2, 15, 2)),
            },
        },
        {
            "name": "GaussianMixture",
            "model": GaussianMixture(),
            "metric": silhouette_score,
            "parameters": {
                "n_components": list(range(2, 30, 2)),
                "covariance_type": ["full", "tied", "diag"],
            },
        },
        {
            "name": "Agglomerative",
            "model": AgglomerativeClustering(),
            "metric": silhouette_score,
            "parameters": {
                "n_clusters": list(range(2, 30, 2)),
                "linkage": ["ward", "complete", "average"],
            },
        },
    ]

    make_plots(embeddings, clusterers)
