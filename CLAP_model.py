from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import torch
import numpy as np
from transformers import ClapModel, ClapProcessor
import os
import subprocess
import json
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# for running on compute cluster
os.environ["PATH"] += (
    os.pathsep + r"/home/ad.msoe.edu/whitcombp/MSOE/Senior_Design/ffmpeg"
)


def get_CLAP_model(load_path=None) -> tuple[ClapModel, ClapProcessor]:
    if load_path is not None:
        model = ClapModel.from_pretrained(load_path)
        processor = ClapProcessor.from_pretrained(load_path)
    else:
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = model.to(device)
    return model, processor


def load_audio_ffmpeg(path, sr=48000):
    """
    Extract mono audio from mp4 using ffmpeg (robust + cross-platform).
    """

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-err_detect",
        "ignore_err",
        "-nostdin",
        "-i",
        path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "f32le",
        "-",
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    audio = np.frombuffer(result.stdout, np.float32)

    if audio is None or audio.size == 0:
        raise ValueError(f"Empty audio in path: {path}")
    return audio, sr


def chunk_audio(audio, sr, chunk_sec, hop_sec):
    """
    Split waveform into overlapping chunks.
    """
    chunk_size = int(chunk_sec * sr)
    hop_size = int(hop_sec * sr)

    chunks = []

    for start in range(0, len(audio) - chunk_size + 1, hop_size):
        chunk = audio[start : start + chunk_size]
        chunks.append(chunk)

    return chunks


def get_clap_embeddings_from_mp4(mp4_paths, model: ClapModel, processor: ClapProcessor):
    """
    Returns CLAP embeddings for a list of mp4 files.

    Args:
        mp4_paths (list[str])
        model (ClapModel)
        processor (ClapProcessor)
        device (str)

    Returns:
        np.ndarray: (N, D)
    """
    model = model.to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for path in tqdm(mp4_paths):
            audio, sr = load_audio_ffmpeg(path)

            chunks = chunk_audio(audio, sr, chunk_sec=10, hop_sec=5)  # overlapping

            chunk_embeds = []
            for chunk in chunks:
                inputs = processor(audio=chunk, sampling_rate=sr, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model.get_audio_features(**inputs).pooler_output
                audio_features = torch.nn.functional.normalize(outputs, dim=-1)
                chunk_embeds.append(audio_features.cpu().numpy()[0])

            aggregated = np.mean(chunk_embeds, axis=0)
            aggregated = aggregated / (np.linalg.norm(aggregated) + 1e-8)
            embeddings.append(aggregated)

    return np.vstack(embeddings)


def diagnose_embeddings(embeddings):
    # From Claude's advice, check for signs of collapsed or low-quality embeddings before clustering.
    X = normalize(np.array(embeddings))

    # 1. Check embedding variance — collapsed embeddings will have near-zero std
    print(f"Mean std per dim (want > 0.05): {X.std(axis=0).mean():.4f}")
    print(f"Dims near-zero std (<0.01): {(X.std(axis=0) < 0.01).sum()}")

    # 2. Check pairwise similarity distribution
    # If genre structure exists, you should see a bimodal distribution

    sim_matrix = cosine_similarity(X)
    upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    print(
        f"Pairwise cosine sim — mean (want mean < 0.95, std > 0.05): {upper.mean():.3f}, std: {upper.std():.3f}"
    )
    # If mean is > 0.95, embeddings are collapsed
    # If std is < 0.05, there's no cluster structure to find

    # 3. PCA explained variance — how many dims carry signal?

    pca = PCA(n_components=min(50, X.shape[1]))
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_dims_90 = np.searchsorted(cumvar, 0.90) + 1
    print(
        f"Dims needed for 90% variance (want > 5): {n_dims_90}"
    )  # want > 5 for genre structure


if __name__ == "__main__":
    clap_model_path = r"/home/ad.msoe.edu/whitcombp/MSOE/PlaylistGenreClassification/training/all_time_favs/10_epochs/finetuned_model"
    model, processor = get_CLAP_model(clap_model_path)

    playlist_dir = r"/home/ad.msoe.edu/whitcombp/MSOE/PlaylistGenreClassification/all time favs_playlist"
    video_files = os.listdir(playlist_dir)
    video_files = [os.path.join(playlist_dir, v) for v in video_files]

    embeddings = get_clap_embeddings_from_mp4(
        mp4_paths=video_files, model=model, processor=processor
    )
    embeddings = embeddings.tolist()
    print("embedding len:", len(embeddings))
    diagnose_embeddings(embeddings)

    with open("embeddings.json", "w") as fp:
        json.dump({"embeddings": embeddings, "files": video_files}, fp, indent=4)
