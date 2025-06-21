def similarity_eval(model_path, tokenizer_path, evals_path, docs_path, num_queries=5, max_len=50):
    '''
    Fixed evaluation function that converts tokens to embeddings before passing to model
    '''
    
    import random
    import torch
    from torch.nn.functional import cosine_similarity
    import pickle
    import models  # Make sure this imports your Towers class
    from load_glove import load_glove  # Import your glove loading function

    # Load model - using embedding_dim from your training params
    embedding_dim = 100  # from your params
    model = models.Towers(embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    words_to_idx = tokenizer['words_to_idx']

    # Load GloVe embeddings (same as used in training)
    glove_path = '/Users/benjipro/MLX/MLX_two_towers/glove_embeddings/glove_embeddings_6B_100d_w2v.txt'
    embeddings = load_glove(glove_path, words_to_idx, embedding_dim=embedding_dim)
    embedding_layer = torch.nn.Embedding.from_pretrained(embeddings, freeze=True)

    # Load eval queries and docs
    with open(evals_path, 'rb') as f:
        eval_data = pickle.load(f)

    # Load all docs
    with open(docs_path, 'rb') as f:
        all_docs = pickle.load(f)

    def embed_text(text, words_to_idx, embedding_layer, max_len=50):
        """Convert text to averaged embeddings tensor"""
        tokens = text.lower().split()  # Basic tokenization
        indices = [words_to_idx.get(token, words_to_idx.get('<UNK>', 0)) for token in tokens[:max_len]]
        
        # Convert to tensor and get embeddings
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        embeddings = embedding_layer(indices_tensor)  # Shape: [seq_len, embedding_dim]
        
        # Average embeddings across sequence length
        avg_embedding = embeddings.mean(dim=0, keepdim=True)  # Shape: [1, embedding_dim]
        
        return avg_embedding

    # Select random eval queries and their correct docs
    selected = random.sample(list(eval_data.items()), num_queries)
    
    # Extract query texts and correct doc IDs
    queries_text = [item[1]['text'] for item in selected]
    correct_doc_ids = [item[1]['docs'][0] for item in selected]
    
    # Get correct doc texts
    correct_docs_text = [all_docs[doc_id] for doc_id in correct_doc_ids]
    
    # Select random docs (excluding correct ones)
    candidate_doc_ids = list(set(all_docs.keys()) - set(correct_doc_ids))
    random_doc_ids = random.sample(candidate_doc_ids, 3 * num_queries)

    print("Starting evaluation...\n")

    for i, (query_text, correct_doc_id) in enumerate(zip(queries_text, correct_doc_ids)):
        # Convert text to embeddings
        query_embedding = embed_text(query_text, words_to_idx, embedding_layer, max_len)
        correct_doc_text = all_docs[correct_doc_id]
        correct_doc_embedding = embed_text(correct_doc_text, words_to_idx, embedding_layer, max_len)
        
        with torch.no_grad():
            # Get embeddings from model towers
            query_emb = model.query_tower(query_embedding)
            correct_doc_emb = model.doc_tower(correct_doc_embedding)
            
            # Calculate similarity with correct doc
            sim_correct = cosine_similarity(query_emb, correct_doc_emb, dim=1).item()
            
            print(f"Query {i+1}: {query_text}")
            print(f"  Correct Doc: {correct_doc_id[:16]}... | Cosine Sim: {sim_correct:.3f}")
            
            # Process 3 random docs for this query
            rand_doc_ids = random_doc_ids[i*3:(i+1)*3]
            
            for j, rand_doc_id in enumerate(rand_doc_ids):
                rand_doc_text = all_docs[rand_doc_id]
                rand_doc_embedding = embed_text(rand_doc_text, words_to_idx, embedding_layer, max_len)
                
                rand_doc_emb = model.doc_tower(rand_doc_embedding)
                sim_rand = cosine_similarity(query_emb, rand_doc_emb, dim=1).item()
                
                print(f"  Random Doc {j+1}: {rand_doc_id[:16]}... | Cosine Sim: {sim_rand:.3f}")
            
            print()  # Empty line between queries


# Additional debugging function to check model behavior
def debug_model_outputs(model_path, tokenizer_path, sample_texts):
    """Debug function to check if model produces different outputs for different inputs"""
    
    import torch
    import pickle
    import models
    from load_glove import load_glove
    
    # Load model and tokenizer
    model = models.Towers(100)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    words_to_idx = tokenizer['words_to_idx']
    
    # Load GloVe embeddings
    glove_path = '/Users/benjipro/MLX/MLX_two_towers/glove_embeddings/glove_embeddings_6B_100d_w2v.txt'
    embeddings = load_glove(glove_path, words_to_idx, embedding_dim=100)
    embedding_layer = torch.nn.Embedding.from_pretrained(embeddings, freeze=True)
    
    def embed_text(text, words_to_idx, embedding_layer, max_len=50):
        tokens = text.lower().split()
        indices = [words_to_idx.get(token, words_to_idx.get('<UNK>', 0)) for token in tokens[:max_len]]
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        embeddings = embedding_layer(indices_tensor)
        avg_embedding = embeddings.mean(dim=0, keepdim=True)
        return avg_embedding
    
    print("Debug: Testing model with different inputs")
    print("=" * 50)
    
    for i, text in enumerate(sample_texts):
        text_embedding = embed_text(text, words_to_idx, embedding_layer)
        
        with torch.no_grad():
            query_emb = model.query_tower(text_embedding)
            doc_emb = model.doc_tower(text_embedding)
        
        print(f"Text {i+1}: {text[:50]}...")
        print(f"  Query embedding mean: {query_emb.mean().item():.6f}")
        print(f"  Doc embedding mean: {doc_emb.mean().item():.6f}")
        print(f"  Query embedding std: {query_emb.std().item():.6f}")
        print(f"  Doc embedding std: {doc_emb.std().item():.6f}")
        print()


# Run your evaluation
model_path = "/Users/benjipro/MLX/MLX_two_towers/checkpoints/2025_06_20__15_15_32.19.350.two.pth"
tokenizer_path = "/Users/benjipro/MLX/MLX_two_towers/corpus/tokeniser.pkl"
evals_path = "/Users/benjipro/MLX/MLX_two_towers/corpus/evals.pkl"
docs_path = "/Users/benjipro/MLX/MLX_two_towers/corpus/docs.pkl"

# First run debug to check model behavior
sample_texts = [
    "how does eating an apple help digestion",
    "caffeine supplements disadvantages",
    "protect yourself from emp attack",
    "good foods for gout sufferers",
    "what is a muscle car"
]

print("Running debug check first...")
debug_model_outputs(model_path, tokenizer_path, sample_texts)

print("\n" + "="*60)
print("Running fixed evaluation...")
print("="*60)
similarity_eval(model_path, tokenizer_path, evals_path, docs_path, num_queries=5)