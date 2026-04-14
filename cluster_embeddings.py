import os
import shutil
import json  # for loading embeddings
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import hdbscan


def cluster_embeddings_to_dirs(
    embeddings,
    file_paths,
    clusterer,
    output_dir="clusters",
):
    assert len(embeddings) == len(file_paths)
    X = normalize(np.array(embeddings))
    labels = clusterer.fit_predict(X)

    # group indices by cluster
    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(i)

    # save as symlinks
    for label, indices in clusters.items():
        cluster_name = "noise" if label == -1 else f"cluster_{label}"
        cluster_dir = os.path.join(output_dir, cluster_name)
        os.makedirs(cluster_dir, exist_ok=True)

        # calculate cluster similarity to sort by
        cluster_embeds = X[indices]
        centroid = cluster_embeds.mean(axis=0, keepdims=True)
        sims = cosine_similarity(cluster_embeds, centroid).flatten()

        # store with similarity appended
        for i, sim in zip(indices, sims):
            path = file_paths[i]
            filename = os.path.basename(path)
            sim_name = f"{sim:.4f} -.- {filename}"
            dst = os.path.join(cluster_dir, sim_name)

            if not os.path.exists(dst):
                os.symlink(os.path.abspath(path), dst)

    return labels


if __name__ == "__main__":
    json_path = "embeddings.json"
    with open(json_path, "r") as fp:
        data = json.load(fp)
    embeddings = data["embeddings"]
    file_paths = data["files"]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=20,
        min_samples=5,
    )

    clusterer = KMeans(n_clusters=30)

    output_path = "clusters"
    shutil.rmtree(output_path)
    labels = cluster_embeddings_to_dirs(
        embeddings,
        file_paths,
        clusterer,
        output_dir=output_path,
    )
