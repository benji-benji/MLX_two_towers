
def similarity_eval(model_path, tokenizer_path, evals_path, docs_path, num_queries=5):
    '''
    (no ranking or retrival, just similarity / dissimilarity)

1. the model is already trained ( weights and biases saved)
2. i want to load up 5 random queries from my eval_queries & 5 associated docs 
3. i then want to load 15 randomly selected queries from my dataset
4. I then want to run the query through the model to get an embedding
5. I then want to run the model on the 20 docs (5 correct docs, and the 15 random docs) 
6. Then for each query, I want to print the correct doc and the 5 random docs
7. I want to compare the cosine similarity of the query & 4 docs (3 random docs & one correct doc)
    '''
    
    import random
    import torch
    from torch.nn.functional import cosine_similarity
    import pickle
    import models  # Make sure this imports your Towers class

    glove_dim = 100
    
    # Load model
    model = models.Towers(glove_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    words_to_idx = tokenizer['words_to_idx']
    
    glove_path = '/Users/benjipro/MLX/MLX_two_towers/glove_embeddings/glove_embeddings_6B_100d_w2v.txt'
    embedding_dim = 100  # or whatever you used in training
    from load_glove import load_glove  # replace with actual import if needed
    embeddings = load_glove(glove_path, words_to_idx, embedding_dim=embedding_dim)
    embedding_layer = torch.nn.Embedding.from_pretrained(embeddings, freeze=True)


    # Load eval queries and docs
    with open(evals_path, 'rb') as f:
        eval_data = pickle.load(f)  # List of (query, correct_doc) pairs

    # Load all docs
    with open(docs_path, 'rb') as f:
        all_docs = pickle.load(f)  # List of doc strings

    # Select num_queries random eval queries and their correct docs
    selected = random.sample(list(eval_data.items()), num_queries)
    print(eval_data)
    print(type(eval_data))
    print(next(iter(eval_data)))
    queries = [v['text'] for k, v in selected]
    correct_docs = [v['docs'][0] for k, v in selected]

    # Select 3*num_queries random docs (excluding correct docs)
    candidate_docs = list(set(all_docs.keys()) - set(correct_docs))
    random_docs = random.sample(candidate_docs, 3 * num_queries)

    # Tokenize and average embeddings (dummy example, replace with your embedding layer)
    def embed(text):
        tokens = [words_to_idx.get(w, words_to_idx['<UNK>']) for w in text.split()]
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        # Assume you have an embedding_layer available
        emb = embedding_layer(tokens_tensor)  # shape: [seq_len, glove_dim]
        avg_emb = emb.mean(dim=0, keepdim=True)  # shape: [1, glove_dim]
        return avg_emb

    for i, (query, correct_doc) in enumerate(zip(queries, correct_docs)):
        query_emb = model.query_tower(embed(query))
        correct_doc_emb = model.doc_tower(embed(correct_doc))
        sim_correct = cosine_similarity(query_emb, correct_doc_emb).item()

        # 3 random docs for this query
        rand_docs = random_docs[i*3:(i+1)*3]
        rand_sims = []
        for rand_doc in rand_docs:
            rand_doc_emb = model.doc_tower(embed(rand_doc))
            sim_rand = cosine_similarity(query_emb, rand_doc_emb).item()
            rand_sims.append((rand_doc, sim_rand))

        print(f"\nQuery {i+1}: {query}")
        print(f"  Correct Doc: {correct_doc[:80]}... | Cosine Sim: {sim_correct:.3f}")
        for j, (rand_doc, sim_rand) in enumerate(rand_sims):
            print(f"  Random Doc {j+1}: {rand_doc[:80]}... | Cosine Sim: {sim_rand:.3f}")


model_path = "/Users/benjipro/MLX/MLX_two_towers/checkpoints/2025_06_20__14_37_33.9.150.two.pth"
tokenizer_path = "/Users/benjipro/MLX/MLX_two_towers/corpus/tokeniser.pkl"
evals_path = "/Users/benjipro/MLX/MLX_two_towers/corpus/evals.pkl"
docs_path = "/Users/benjipro/MLX/MLX_two_towers/corpus/docs.pkl"
similarity_eval(model_path, tokenizer_path, evals_path, docs_path, num_queries=5)