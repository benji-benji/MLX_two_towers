import torch
import torch.nn as nn
import pickle
import numpy as np
from models import Towers
from load_glove import load_glove
from sklearn.metrics.pairwise import cosine_similarity

EMBED_DIM = 100

# Load tokenizer
with open('./corpus/tokeniser.pkl', 'rb') as f:
    tkns = pickle.load(f)
words_to_idx = tkns['words_to_idx']

# Load doc embeddings and IDs
doc_embeddings = np.load('./corpus/doc_embeddings.final.npy')
with open('./corpus/doc_ids.final.pkl', 'rb') as f:
    doc_ids = pickle.load(f)

# Load docs and eval queries
with open('./corpus/docs.pkl', 'rb') as f:
    docs = pickle.load(f)
with open('./corpus/evals.pkl', 'rb') as f:
    evals = pickle.load(f)

# Model and Embedding Layer
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
glove_path = '/Users/aparna/Documents/CollabWeek2/MLX_two_towers/corpus/glove.6B.100d.txt'
embeddings = load_glove(glove_path, words_to_idx, embedding_dim=EMBED_DIM)
embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True)
model = Towers(glove_dim=EMBED_DIM).to(dev)
model.load_state_dict(torch.load('./checkpoints/final.two.pth', map_location=dev))
model.eval()

def tokenize(text, words_to_idx):
    return torch.tensor([words_to_idx.get(w, words_to_idx['<UNK>']) for w in text.lower().split() if w.strip()], dtype=torch.long)

def retrieve_top_k(query_text, k=5):
    toks = tokenize(query_text, words_to_idx).unsqueeze(0).to(dev)
    with torch.no_grad():
        emb = embedding_layer(toks)
        avg_emb = emb.mean(dim=1)
        query_emb = model.query_tower(avg_emb).cpu().numpy()
    sims = cosine_similarity(query_emb, doc_embeddings).flatten()
    top_k_idx = np.argsort(-sims)[:k]
    top_k_doc_hashes = [doc_ids[i] for i in top_k_idx]
    top_k_scores = sims[top_k_idx]
    return top_k_doc_hashes, top_k_scores

def evaluate(eval_dict, k=5, verbose=True):
    hit_count = 0
    total = len(eval_dict)
    for qid, entry in eval_dict.items():
        q_text = entry['text']
        gt_docs = set(entry['docs'])
        top_k_hashes, top_k_scores = retrieve_top_k(q_text, k=k)
        
        if verbose:
            print(f"\nEval Query: {q_text}")
            print("Ground truth doc hashes:", list(gt_docs))
            print("Top-{} retrieved:".format(k))
            for i, (doc_hash, score) in enumerate(zip(top_k_hashes, top_k_scores), 1):
                print(f"{i}. [{score:.3f}] {doc_hash} | {docs[doc_hash][:120].replace('\n',' ')}...")

        hits = gt_docs.intersection(top_k_hashes)
        if hits:
            hit_count += 1
            if verbose:
                print("Hit! Ground-truth doc found in Top-{}.".format(k))
        else:
            if verbose:
                print("Miss. No ground-truth doc found in Top-{}.".format(k))
        if verbose:
            print('-'*40)
    hit_at_k = hit_count / total if total > 0 else 0.0
    print(f"\nFinal Hit@{k}: {hit_count}/{total} = {hit_at_k:.3f}")
    return hit_at_k

# ---- Run evaluation ----
if __name__ == "__main__":
    score = evaluate(evals, k=5, verbose=True)
    print(f"\nHit@5 (accuracy): {score:.3f}")
