import torch
import torch.nn as nn
import numpy as np

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

class PretrainedWordEmbeddings: 
    
    def __init__(self, path, vocab = dict):
        # takes two parameters path to embeddings and vocab dictionary 
        self.emb_dim = 100
        self.vocab = vocab 
        self.embeddings = self.load_embeddings(path, vocab)
        
    def load_embeddings(self,path,vocab):
        # empty dict to store words in
        embeddings_index ={}
        
        # Opens the embeddings file (expected format: word followed by space-separated numbers)
        with open(path, 'r', encoding='utf8') as f:
            # for each line
            for line in f:
                # strip whitespace , split each space
                tokens = line.rstrip().split()
                # First token is the word, remaining tokens are the vector components
                word = tokens[0]
                # Converts vector components to numpy array of floats
                vec = torch.tensor([float(tokens) for tokens in tokens[1:]])
                # stores embeddings for words that exist in vocab
                if word in vocab:
                    embeddings_index[word] = vec
        # for words not found in GloVe, assign random vectors 
        
        # create embedding matrix for all vocabulary words
        vocab_size = len(vocab)
        emb_matrix = torch.zeros((vocab_size, self.emb_dim))
        
        # for each word in vocab: uses loaded embedding if available, otherwise generates random vector
        for word, idx in vocab.items():
            if word in embeddings_index:
                emb_matrix[idx] = embeddings_index[word]
            else: 
                emb_matrix[idx] = torch.rand(self.emb_dim) * 0.01
                
            
            # Random vectors are sampled from normal distribution with standard deviation 0.6
            #vec = embeddings.get(word, np.random.normal(scale=0.6, size=(self.emb_dim)))            
            
            # add non-vocab words to embedding matrix
            #emb_matrix.append(vec)
        # return torch tensor     
        #single_array = np.array(emb_matrix)
        #return torch.tensor(single_array,dtype=torch.float)
        emb_layer = nn.Embedding.from_pretrained(emb_matrix, freeze=False)
        return emb_layer
    
    def encode(self, text):
        # method to encode text 
        
        # split text into words, convert to lowercase
        idxs = [self.vocab[w] for w in text.lower().split() if w in self.vocab]
        # create zero matrix for non-vocab words
        if not idxs:
            return torch.zeros(self.emb_dim)
        
        # map each word to its vocabulary index (only includes words present in vocab)
        idxs_tensor = torch.tensor(idxs, dtype=torch.long)
        vectors = self.embeddings(idxs_tensor)
        
        #return the mean of all word vectors 
        return vectors.mean(dim=0)
        
        
        
        
                
    
    

