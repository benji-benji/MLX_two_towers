import os
import torch 
import torch.nn as nn
import torch.optim as optim
from embeddings import ToyWordEmbeddings, PretrainedWordEmbeddings
from data import queries, docs, triples, mini_real_queries_list, mini_real_passages_list

# create a set containing all words in both queries and docs 
# join the sets of srings into a new string
# the "" empty quotes contain the character to use to join
# split the resulting long string accoring to .split()
# when empty split() defaults to sep = whitespace
# this is essentially our vocabulary
all_words = set(" ".join(queries + docs).split())

# sense check, print all words in vocab
print(all_words)

# sense check, print some items from set 
for i, x in enumerate(all_words):
    if i in (2,5,6):
        print(x)

# create dictionary using curly brackets
# i is the index, w is the word 
# this line creates a dictionary of words = numbers from all_words dataset 
# after sorting alphabetically
vocab = {w: i for i, w in enumerate(sorted(all_words))}

# toy embeddings: 
# emb_dim = 10 # must match TowerOne's input!
# embeddings = ToyWordEmbeddings(vocab, emb_dim)

# Pretrained embeddings:
emb_dim = 100
embedding_path = "/Users/benjipro/MLX/MLX_two_towers/glove.6B.100d.word2vec.embeddings.txt"
print ("File exists:", os.path.exists(embedding_path))
embeddings = PretrainedWordEmbeddings(embedding_path, vocab)

print(type(embeddings))

class Query_Tower(torch.nn.Module):
# nn.module is a fundamental class in PyTorch used to create custom neural network architectures.
# Define a neural network class for the query encoder
# this is a single-layer perceptron
# toy version works with this because it is lineraly seperable

#   def __init__(self):
#        super().__init__()
#        # a linear layer (fc) that maps a 10-dimensional input vector to a 1-dimensional output.
#        self.fc = torch.nn.Linear(emb_dim,1)

    def __init__(self):
        super(Query_Tower, self).__init__()
        self.fc = nn.Linear(emb_dim, 64)
        self.out = nn.Linear(64, 32)
        
#    def forward(self, x):
#        # defines how input x is processed, simply passed through linear layer 
#        x = self.fc(x)
#        return x

    def forward(self, x):
        
        return self.out(torch.relu(self.fc(x)))

        
    
class Doc_Tower(torch.nn.Module):
# Define a neural network class for the doc encoder

#    def __init__(self):
#        super(Doc_Tower, self).__init__()
#        # a linear layer (fc) that maps a 10-dimensional input vector to a 1-dimensional output.

#        self.fc = torch.nn.Linear(10,1)

    def __init__(self):
        super(Doc_Tower, self).__init__()
        self.fc = nn.Linear(emb_dim, 64)
        self.out = nn.Linear(64, 32)
        
#    def forward(self, x):
#        # defines how input x is processed, simply passed through linear layer 
#        x = self.fc(x)
#        return x
    def forward(self, x):
        
        return self.out(torch.relu(self.fc(x)))

# instantiates both models
Query_Tower = Query_Tower()
Doc_Tower = Doc_Tower()


# Creates dummy input vectors
query_dummy = torch.randn(1, 100)
doc_positive_dummy = torch.randn(1, 100)
doc_negative_dummy = torch.randn(1, 100)

# Passes the vectors through the respective towers
Query = Query_Tower(query_dummy)
Doc_Positive = Doc_Tower(doc_positive_dummy)
Doc_Negative = Doc_Tower(doc_negative_dummy)

# Computes cosine similarity
# Since these are 1D outputs,
# cosine similarity will return either 1 or -1, 
# depending onwhether the scalars have the same or opposite sign.
distance_positive = torch.nn.functional.cosine_similarity(Query, Doc_Positive)
distance_negative = torch.nn.functional.cosine_similarity(Query, Doc_Negative)

# Measures how much closer the positive doc is than the negative doc to the query.
distance_difference = distance_positive - distance_negative

# Set a margin for contrastive learning. 
# We want the positive to be at least 0.
# more similar than the negative
distance_margin = torch.tensor(0.2)

# Computes a margin-based loss:
loss = torch.max(torch.tensor(0.0), distance_margin - distance_difference)
loss.backward()

print("Loss:", loss.item())
print(distance_positive)

# TRAINING LOOP 

optimizer = optim.Adam(list(embeddings.embeddings.parameters()) +
                      list(Query_Tower.parameters()) +
                      list(Doc_Tower.parameters()), lr=0.01)

for epoch in range(15):
    total_loss = 0
    for (q_idx, d_pos_idx, d_neg_idx) in triples:
        q_vec = embeddings.encode(queries[q_idx])
        d_pos_vec = embeddings.encode(docs[d_pos_idx])
        d_neg_vec = embeddings.encode(docs[d_neg_idx])

        # expand dims to shape (1, emb_dim) as in example
        d_pos_vec = d_pos_vec.unsqueeze(0)
        d_neg_vec = d_neg_vec.unsqueeze(0)
        q_vec = q_vec.unsqueeze(0)

        out_doc = Query_Tower(d_pos_vec)
        out_neg = Query_Tower(d_neg_vec)
        out_query = Doc_Tower(q_vec)

        sim_pos = nn.functional.cosine_similarity(out_query, out_doc)
        sim_neg = nn.functional.cosine_similarity(out_query, out_neg)

        margin = 0.1
        loss = torch.relu(margin + sim_neg - sim_pos).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch} loss: {total_loss/len(triples):.4f}")
    
    
def retrieve(query):
    q_vec = embeddings.encode(query).unsqueeze(0)
    out_query = Doc_Tower(q_vec)
    scores = []
    for doc in docs:
        d_vec = embeddings.encode(doc).unsqueeze(0)
        out_doc = Doc_Tower(d_vec)
        sim = nn.functional.cosine_similarity(out_query, out_doc)
        scores.append(sim.item())
    return scores

for i, q in enumerate(queries):
    sims = retrieve(q)
    print(f"\nQuery: {q}")
    for j, s in enumerate(sims):
        print(f"  Doc {j}: Score: {s:.3f}")    