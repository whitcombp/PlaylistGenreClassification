import torch
import numpy as np
import torchaudio
from transformers import ClapModel, ClapProcessor
import os
import subprocess
import json
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_CLAP_model() -> tuple[ClapModel, ClapProcessor]:
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = model.to(device)
    return model, processor


def load_audio_ffmpeg(path, sr=48000):
    """
    Extract mono audio from mp4 using ffmpeg (robust + cross-platform).
    """

    cmd = ["ffmpeg", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    audio = np.frombuffer(result.stdout, np.float32)
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


def get_clap_embeddings_from_mp4(mp4_paths, model, processor):
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

                outputs = model.get_audio_features(**inputs)
                # shape: (1, CLAP windows, 2, embedding size)
                audio_features = outputs.last_hidden_state
                # mean non-feature dims, (1, embedding size)
                audio_features = audio_features.mean(dim=(1, 2))
                # normalize
                audio_features = torch.nn.functional.normalize(audio_features, dim=-1)
                # append chunk and repeat
                chunk_embeds.append(audio_features.cpu().numpy()[0])

            aggregated_embedding = np.median(chunk_embeds, axis=0)
            embeddings.append(aggregated_embedding)

    return np.vstack(embeddings)


if __name__ == "__main__":
    model, processor = get_CLAP_model()

    playlist_dir = r"C:\Users\whitcombp\Documents\Personal\Python\Actually Kinda Somewhat Decent Acceptable Songs_playlist"
    video_files = os.listdir(playlist_dir)
    video_files = [os.path.join(playlist_dir, v) for v in video_files]

    embeddings = get_clap_embeddings_from_mp4(
        mp4_paths=video_files, model=model, processor=processor
    )
    embeddings = embeddings.tolist()
    print("embedding len:", len(embeddings))
    print("first embedding:\n", embeddings[0])

    with open("embeddings.json", "w") as fp:
        json.dump({"embeddings": embeddings, "files": video_files}, fp, indent=4)
