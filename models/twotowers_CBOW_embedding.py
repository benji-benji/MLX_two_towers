import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from itertools import chain
from tqdm import tqdm
from torchtext.data.utils import get_tokenizer
import requests
import numpy as np
import matplotlib.pyplot as plt
import json

# Ground truth word pairs for similarity evaluation
ground_truth_pairs = [
    ("cat", "dog"),
    ("car", "bus"),
    ("apple", "orange"),
    ("cat", "car"),       # unrelated
    ("music", "song"),
    ("king", "queen"),
    ("table", "banana"),  # unrelated
]


# Define expected similarity ranges for each pair
expected_ranges = {
    ("cat", "dog"): (0.4, 0.7),
    ("car", "bus"): (0.3, 0.6),
    ("apple", "orange"): (0.4, 0.7),
    ("cat", "car"): (0.0, 0.2),        # unrelated
    ("music", "song"): (0.3, 0.6),
    ("king", "queen"): (0.4, 0.7),
    ("table", "banana"): (0.0, 0.2),   # unrelated
}

def evaluate_ground_truth_pairs(model, vocab, ground_truth_pairs):
    print("\n--- Ground Truth Pair Similarity ---")
    emb_norm = F.normalize(model.embeddings.weight, dim=1)
    for w1, w2 in ground_truth_pairs:
        if w1 in vocab and w2 in vocab:
            idx1, idx2 = vocab[w1], vocab[w2]
            sim = torch.dot(emb_norm[idx1], emb_norm[idx2]).item()
            # Get expected range, default to (0,0) if not found
            expected = expected_ranges.get((w1, w2), (0, 0))
            in_range = expected[0] <= sim <= expected[1]
            print(f"Cosine similarity between '{w1}' and '{w2}': {sim:.4f} | Expected: {expected} | TRUE" if in_range else
                  f"Cosine similarity between '{w1}' and '{w2}': {sim:.4f} | Expected: {expected}")
        else:
            print(f"One or both words not in vocab: '{w1}', '{w2}'")


total_num_tokens = 4_000_000  # total number of tokens to use
batch_size = 256  # batch size for training
demensions = 128  # embedding dimension
learning_rate = 0.003  # learning rate for optimizer
window_size = 2
number_of_epochs = 5  # number of epochs for training
min_count = 20

url = "https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8"
response = requests.get(url)
text = response.text
tokenizer = get_tokenizer("basic_english")
tokens_list = tokenizer(text)  # tokenize entire text at once
counter = Counter(tokens_list)  # print first 10 tokens for verification
sentences = tokens_list[:total_num_tokens]  # use first 80,000 tokens as sentences

#print(sentences[:100])

def create_windows(tokens, window_size=window_size):
    """Create context-target pairs using sliding windows"""
    data = []
    for i in range(window_size, len(tokens) - window_size):
        context = tokens[i-window_size : i] + tokens[i+1 : i+window_size+1]
        target = tokens[i]
        data.append((context, target))
    return data

# Create windows
window_size = window_size  # 2 words on each side
data = create_windows(tokens_list[:total_num_tokens], window_size)  # Use first 80k tokens

# Build vocabulary
all_words = [word for context, target in data for word in context + [target]]
word_counts = Counter(all_words)
min_count = 20  # Set your min_count value here
filtered_words = {word for word, count in word_counts.items() if count >= min_count}

# Create vocabulary mapping
vocab = {word: i for i, word in enumerate(filtered_words)}
index_to_word = {i: w for w, i in vocab.items()}
vocab_size = len(vocab)
print(f"Vocabulary size after filtering: {vocab_size}")

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)
    

# Filter data to only include words from the vocabulary
filtered_data = []
for context, target in data:
    # Only keep contexts where all words are in vocabulary
    if all(word in filtered_words for word in context) and target in filtered_words:
        filtered_data.append((context, target))

# Index the FILTERED data - THIS IS THE CRITICAL FIX
indexed_data = [([vocab[w] for w in context], vocab[target]) 
               for context, target in filtered_data]  # Use filtered_data here
data = indexed_data

class CBOWDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(
            target, dtype=torch.long
        )


dataset = CBOWDataset(data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Model definition
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idxs):
        embeds = self.embeddings(context_idxs)
        avg_embeds = embeds.mean(dim=1)
        out = self.linear(avg_embeds)
        return out


def print_most_similar_words(query_word, model, vocab, index_to_word, top_k=5):
    if query_word not in vocab:
        print(f"'{query_word}' not in vocab.")
        return
    query_idx = vocab[query_word]
    query_vec = model.embeddings.weight[query_idx]
    emb_norm = F.normalize(model.embeddings.weight, dim=1)
    query_vec = F.normalize(query_vec, dim=0)
    cos_sim = torch.matmul(emb_norm, query_vec)
    topk = torch.topk(cos_sim, top_k + 1)
    top_indices = topk.indices.tolist()
    print(f"\nTop {top_k} words similar to '{query_word}':")
    count = 0
    for idx in top_indices:
        word = index_to_word[idx]
        if word != query_word:
            print(f"  {word} (score: {cos_sim[idx]:.4f})")
            count += 1
        if count == top_k:
            break





# Training
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device selection cuda if available or apple gpu if not, else cpu
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") if not torch.cuda.is_available() else device
print(f"Using device: {device}")

embedding_dim = 200
model = CBOWModel(vocab_size, embedding_dim).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epoch_losses = []  # Store losses for plotting later

for epoch in range(number_of_epochs):
    total_loss = 0
    model.train()
    for context_idxs, target_idx in dataloader:
        context_idxs, target_idx = context_idxs.to(device), target_idx.to(device)
        optimizer.zero_grad()
        out = model(context_idxs)
        loss = loss_function(out, target_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    model.eval()
    epoch_as_int = epoch + 1
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "epoch_as_int" + "final_embedding_epoch.pth")  
    if (epoch + 1) %2  == 0:
        for test_word in ["american", "computer", "table"]:
            print_most_similar_words(test_word, model, vocab, index_to_word, top_k=3)
        evaluate_ground_truth_pairs(model, vocab, ground_truth_pairs)
    
epochs = list(range(1, number_of_epochs + 1))
plt.plot(epochs, epoch_losses, marker='o')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()



