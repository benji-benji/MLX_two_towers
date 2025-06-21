import torch
import pickle
import torch.nn as nn
import models
from load_glove import load_glove
import numpy as np

# Hyperparameters (should match training)
EMBED_DIM = 100
BATCH_SIZE = 256
CHECKPOINT_PATH = './checkpoints/final.two.pth'

# Load tokenizer
with open('./corpus/tokeniser.pkl', 'rb') as f:
    tkns = pickle.load(f)
words_to_idx = tkns['words_to_idx']

# Load docs
with open('./corpus/docs.pkl', 'rb') as f:
    docs = pickle.load(f)

# Device
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load GloVe and model
glove_path = '/Users/aparna/Documents/CollabWeek2/MLX_two_towers/corpus/glove.6B.100d.txt'
embeddings = load_glove(glove_path, words_to_idx, embedding_dim=EMBED_DIM)
embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True)

model = models.Towers(glove_dim=EMBED_DIM).to(dev)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=dev))
model.eval()

def tokenize(text, words_to_idx):
    return torch.tensor([words_to_idx.get(w, words_to_idx['<UNK>']) for w in text.lower().split() if w.strip()], dtype=torch.long)

# Precompute doc embeddings (batched)
doc_ids = list(docs.keys())
doc_texts = [docs[d] for d in doc_ids]
doc_embeddings = []

with torch.no_grad():
    for i in range(0, len(doc_texts), BATCH_SIZE):
        batch_texts = doc_texts[i:i+BATCH_SIZE]
        batch_tokens = [tokenize(t, words_to_idx) for t in batch_texts]
        maxlen = max(len(t) for t in batch_tokens)
        batch_padded = [torch.cat([t, torch.zeros(maxlen - len(t), dtype=torch.long)]) if len(t) < maxlen else t for t in batch_tokens]
        batch_tensor = torch.stack(batch_padded).to(dev)  # [batch, seq_len]

        # Create word embeddings using your embedding_layer
        emb = embedding_layer(batch_tensor)  # [batch, seq_len, emb_dim]
        avg_emb = emb.mean(dim=1)           # [batch, emb_dim]

        # Pass through the document tower
        batch_emb = model.doc_tower(avg_emb)  # [batch, emb_dim]
        doc_embeddings.append(batch_emb.cpu())
    doc_embeddings = torch.cat(doc_embeddings, dim=0)

# Save as numpy for quick reload later
np.save('./corpus/doc_embeddings.final.npy', doc_embeddings.numpy())
with open('./corpus/doc_ids.final.pkl', 'wb') as f:
    pickle.dump(doc_ids, f)

print(f"Saved {len(doc_ids)} document embeddings to ./corpus/doc_embeddings.final.npy")
