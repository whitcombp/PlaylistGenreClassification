import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from transformers import ClapModel, ClapProcessor
from tqdm import tqdm
import random
from CLAP_model import chunk_audio, load_audio_ffmpeg, get_CLAP_model

device = "cuda" if torch.cuda.is_available() else "cpu"
_chunk_cache = {}


def sample_positive_pair(audio, sr):
    chunks = chunk_audio(audio, sr, chunk_sec=10, hop_sec=5)
    c1, c2 = random.sample(chunks, 2)
    return c1, c2


def sample_positive_pair_precomputed(audio_path):
    if audio_path not in _chunk_cache:
        audio = np.load(audio_path)
        chunks = chunk_audio(audio, sr=48000, chunk_sec=10, hop_sec=5)
        _chunk_cache[audio_path] = chunks

    chunks = _chunk_cache[audio_path]
    c1, c2 = random.sample(chunks, 2)
    return c1, c2


def get_embedding_batch(audios, sr, model: ClapModel, processor: ClapProcessor):
    inputs = processor(
        audio=audios,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.get_audio_features(**inputs)
    embedding = outputs.pooler_output
    embedding = torch.nn.functional.normalize(embedding, dim=-1)
    return embedding


def contrastive_loss(z1, z2, temperature=0.1):
    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    # mask to prevent trivial self-matching
    mask = torch.eye(sim.size(0), device=sim.device).bool()
    sim.masked_fill_(mask, -9e15)

    labels = torch.arange(batch_size).to(z.device)
    labels = torch.cat([labels, labels], dim=0)

    loss = torch.nn.CrossEntropyLoss()(sim, labels)
    return loss


def graph_training_metrics(train_loss, train_alignment, train_uniformity, title):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].plot(train_loss)
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Value")

    axs[1].plot(train_alignment)
    axs[1].set_title("Alignment")
    axs[1].set_xlabel("Epochs")

    axs[2].plot(train_uniformity)
    axs[2].set_title("Uniformity")
    axs[2].set_xlabel("Epochs")

    plt.suptitle(title)
    plt.savefig(title)
    plt.close()


def debug_embeddings(z1, z2):
    with torch.no_grad():
        print("\n=== EMBEDDING DEBUG ===")

        print("z1 shape:", z1.shape)
        print("z2 shape:", z2.shape)

        # norms (should be ~1.0 after normalize)
        z1_norms = z1.norm(dim=1)
        z2_norms = z2.norm(dim=1)

        print("z1 norms:", z1_norms)
        print("z2 norms:", z2_norms)

        # similarity between positive pairs
        pos_sim = torch.sum(z1 * z2, dim=1)
        print("positive similarity:", pos_sim)

        # similarity across batch (detect collapse)
        sim_matrix = torch.matmul(z1, z1.T)
        print("intra-batch similarity (z1 vs z1):")
        print(sim_matrix)

        print("mean sim:", sim_matrix.mean().item())
        print("std sim:", sim_matrix.std().item())

        print("========================\n")


def train(
    train_video_files: list,
    model: ClapModel,
    processor: ClapProcessor,
    num_epochs: int,
    batch_size: int,
    is_already_audio=False,
):
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    model.audio_projection.requires_grad_(True)

    optimizer = torch.optim.Adam(model.audio_projection.parameters(), lr=1e-4)
    train_loss = []
    train_alignment = []
    train_uniformity = []
    for _ in range(num_epochs):
        epoch_loss = 0.0
        epoch_alignment = 0.0
        epoch_uniformity = 0.0
        num_batches = len(train_video_files) // batch_size
        for _ in tqdm(range(num_batches)):
            batch_pair_1 = []
            batch_pair_2 = []
            for _ in range(batch_size):
                path = random.choice(train_video_files)
                if not is_already_audio:
                    audio, sr = load_audio_ffmpeg(path)
                    pair_1, pair_2 = sample_positive_pair(audio, sr)
                else:
                    audio = np.load(path)
                    sr = 48000  # all preprocessed files have the same sr
                    pair_1, pair_2 = sample_positive_pair(audio, sr)

                batch_pair_1.append(pair_1)
                batch_pair_2.append(pair_2)

            z1 = get_embedding_batch(batch_pair_1, sr, model, processor)
            z2 = get_embedding_batch(batch_pair_2, sr, model, processor)
            # debug_embeddings(z1, z2)
            loss = contrastive_loss(z1, z2)
            alignment = torch.mean(torch.sum((z1 - z2) ** 2, dim=1))
            uniformity = torch.log(torch.mean(torch.exp(-2 * torch.pdist(z1, p=2))))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_alignment += alignment.item() / num_batches
            epoch_uniformity += uniformity.item() / num_batches
        train_loss.append(epoch_loss)
        train_alignment.append(epoch_alignment)
        train_uniformity.append(epoch_uniformity)

    title = f"Train results after {num_epochs} epochs"
    graph_training_metrics(train_loss, train_alignment, train_uniformity, title)

    return model


if __name__ == "__main__":
    model, processor = get_CLAP_model()
    video_files_path = r"/home/ad.msoe.edu/whitcombp/MSOE/PlaylistGenreClassification/Actually Kinda Somewhat Decent Acceptable Songs_playlist"
    # We'll just use all the video files to fully fit the model on our data
    # this should give us the most accurate embeddings
    train_video_files = os.listdir(video_files_path)
    train_video_files = [os.path.join(video_files_path, f) for f in train_video_files]
    # perform ffmpeg operation before training
    ffmpeg_dir = r"/home/ad.msoe.edu/whitcombp/MSOE/PlaylistGenreClassification/playlist_after_ffmpeg"
    if not os.path.isdir(ffmpeg_dir):
        os.makedirs(ffmpeg_dir)
        for f in tqdm(train_video_files, desc="Precomputing FFMPEG audio array"):
            audio, _ = load_audio_ffmpeg(f)
            save_path = os.path.basename(f).replace(".mp4", ".npy")
            np.save(os.path.join(ffmpeg_dir, save_path), audio)
    # get all paths to pass to training, same as before
    train_numpy_files = os.listdir(ffmpeg_dir)
    train_numpy_files = [os.path.join(ffmpeg_dir, f) for f in train_numpy_files]
    model = train(
        train_numpy_files,
        model,
        processor,
        num_epochs=100,
        batch_size=32,
        is_already_audio=True,
    )
    model.save_pretrained("finetuned_model")
    processor.save_pretrained("finetuned_model")
