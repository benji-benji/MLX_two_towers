import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
from models import Towers
from load_glove import load_glove
from sklearn.metrics.pairwise import cosine_similarity

# ---- Constants ----
EMBED_DIM = 100
DOC_EMB_PATH = './corpus/doc_embeddings.final.npy'
DOC_ID_PATH = './corpus/doc_ids.final.pkl'
DOC_TEXT_PATH = './corpus/docs.pkl'
TOKENIZER_PATH = './corpus/tokeniser.pkl'
CHECKPOINT_PATH = './checkpoints/final.two.pth'
GLOVE_PATH = '/Users/aparna/Documents/CollabWeek2/MLX_two_towers/corpus/glove.6B.100d.txt'

# ---- Load assets ----
@st.cache_resource(show_spinner="Loading model & data ...")
def load_assets():
    # Tokenizer
    with open(TOKENIZER_PATH, 'rb') as f:
        tkns = pickle.load(f)
    words_to_idx = tkns['words_to_idx']
    # Doc encodings and IDs
    doc_embeddings = np.load(DOC_EMB_PATH)
    with open(DOC_ID_PATH, 'rb') as f:
        doc_ids = pickle.load(f)
    # Doc text
    with open(DOC_TEXT_PATH, 'rb') as f:
        docs = pickle.load(f)
    # Model and embedding layer
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings = load_glove(GLOVE_PATH, words_to_idx, embedding_dim=EMBED_DIM)
    embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True)
    model = Towers(glove_dim=EMBED_DIM).to(dev)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=dev))
    model.eval()
    return words_to_idx, doc_embeddings, doc_ids, docs, model, dev, embedding_layer

words_to_idx, doc_embeddings, doc_ids, docs, model, dev, embedding_layer = load_assets()

# ---- Tokenize ----
def tokenize(text, words_to_idx):
    return torch.tensor(
        [words_to_idx.get(w, words_to_idx['<UNK>']) for w in text.lower().split() if w.strip()],
        dtype=torch.long
    )

# ---- Retrieval function ----
def retrieve_top_k(query_text, k=3):
    toks = tokenize(query_text, words_to_idx).unsqueeze(0).to(dev)
    with torch.no_grad():
        emb = embedding_layer(toks)        # [1, seq_len, emb_dim]
        avg_emb = emb.mean(dim=1)          # [1, emb_dim]
        query_emb = model.query_tower(avg_emb).cpu().numpy()  # [1, emb_dim]
    sims = cosine_similarity(query_emb, doc_embeddings).flatten()
    top_k_idx = np.argsort(-sims)[:k]
    top_k_doc_hashes = [doc_ids[i] for i in top_k_idx]
    top_k_scores = sims[top_k_idx]
    top_k_texts = [docs[h] for h in top_k_doc_hashes]
    return list(zip(top_k_doc_hashes, top_k_scores, top_k_texts))

# ---- Streamlit UI ----
st.title("MLX Two-Tower Document Retriever")

query = st.text_input("Enter your query:")

if query:
    st.write("### Top 3 most relevant documents:")
    top_docs = retrieve_top_k(query, k=3)
    for i, (doc_hash, score, doc_text) in enumerate(top_docs, 1):
        st.markdown(f"**{i}. [Score: {score:.3f}] Doc Hash:** `{doc_hash}`")
        st.write(doc_text[:400] + ("..." if len(doc_text) > 400 else ""))
        st.markdown("---")
