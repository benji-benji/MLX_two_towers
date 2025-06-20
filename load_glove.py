import torch
import numpy as np

UNK_TOKEN = "<UNK>"

def load_glove(glove_path, word_to_idx, embedding_dim=100):
    # Initialize with zeros
    embeddings = np.zeros((len(word_to_idx.values()) + 1, embedding_dim))

    # Load GloVe vectors
    with open(glove_path, "r") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in word_to_idx:
                vector = np.array([float(x) for x in parts[1:]])
                embeddings[word_to_idx[word]] = vector

    # Handle UNK token (average of all vectors)
    unk_idx = word_to_idx[UNK_TOKEN]
    embeddings[unk_idx] = np.mean(embeddings[1:], axis=0)  # skip PAD

    return torch.tensor(embeddings, dtype=torch.float32)
