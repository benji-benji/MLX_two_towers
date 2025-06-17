import torch
import torch.nn as nn

class ToyWordEmbeddings:

# a class for creating simple dummy embeddings
# which uses a vocabulary and  
  
    def __init__(self, vocab, emb_dim):
        # store vocab
        self.vocab = vocab
        # create empty shape same size as embedding 
        # i.e dummy for the embeddings a word2vec model outputs 
        self.emb = nn.Embedding(len(vocab), emb_dim)
    
    def encode(self, text):
    # method to take some text and embed it
        
        # split input string into words
        # create a variable for the indexs of each token
        # word that appears in vocab
        idxs = [self.vocab[word] for word in text.split() if word in self.vocab]
        # if no words are in vocab return zero vector of correct dim
        # ignores out of vocab words
        if not idxs:
            return torch.zeros(self.emb.embedding_dim)
        
        #convert the list of idexes to a tensor 
        tensor = torch.tensor(idxs)
        
        # feed the tensor into the embedding layer 
        # average the total 
        return self.emb(tensor).mean(dim=0)


    