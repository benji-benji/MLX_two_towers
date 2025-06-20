import torch
import random
import pickle
from torch.utils.data import Dataset

class Triplets(Dataset):
  def __init__(self, embs, tkns):
    self.embs = embs
    self.tkns = tkns
    self.qrys = pickle.load(open('./corpus/qrys.pkl', 'rb'))
    self.docs = pickle.load(open('./corpus/docs.pkl', 'rb'))
    self.q_keys = list(self.qrys.keys())
    self.d_keys = list(self.docs.keys())
    with open('./corpus/tokeniser.pkl', 'rb') as f: tkns = pickle.load(f)
    self.words_to_idx = tkns['words_to_idx']

  def __len__(self):
    return len(self.qrys)

  def __getitem__(self, idx):
    qry = self.qrys[self.q_keys[idx]]
    pos = self.docs[qry['docs'][0]]
    neg = self.docs[random.choice(self.d_keys)]
    qry = self.to_emb(qry['text'])
    pos = self.to_emb(pos)
    neg = self.to_emb(neg)
    return qry, pos, neg

  def to_emb(self, text):
    text = self.preprocess(text)
    tkns = [self.tkns[t] for t in text if t in self.tkns]
    #if len(tkns) == 0: return
    if len(tkns) == 0:
        return torch.zeros(self.embs.embedding_dim)
    #tkns = torch.tensor(tkns).to('cuda:0')
    tkns = torch.tensor(tkns).to('cpu')
    embs = self.embs(tkns)
    return embs.mean(dim=0)

  def preprocess(self, text):
    text = text.lower()
    text = text.replace('.',  ' <PERIOD> ')
    text = text.replace(',',  ' <COMMA> ')
    text = text.replace('"',  ' <QUOTATION_MARK> ')
    text = text.replace('“',  ' <QUOTATION_MARK> ')
    text = text.replace('”',  ' <QUOTATION_MARK> ')
    text = text.replace(';',  ' <SEMICOLON> ')
    text = text.replace('!',  ' <EXCLAMATION_MARK> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace('(',  ' <LEFT_PAREN> ')
    text = text.replace(')',  ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace(':',  ' <COLON> ')
    text = text.replace("'",  ' <APOSTROPHE> ')
    text = text.replace("’",  ' <APOSTROPHE> ')
    return text.split()

if __name__ == "__main__":
    # Dummy check: assume you have loaded emb layer and tokeniser already
    # Replace the following lines with your actual embedding layer and tokeniser
    import torch.nn as nn
    
    # Dummy embeddings for testing — use your real model in practice
    vocab_size = 10000
    embedding_dim = 100
    dummy_emb_layer = nn.Embedding(vocab_size, embedding_dim).to('cpu')

    # Dummy tokeniser mapping
    dummy_tokeniser = {str(i): i for i in range(vocab_size)}
    with open('./corpus/tokeniser.pkl', 'wb') as f:
        pickle.dump({"words_to_idx": dummy_tokeniser}, f)

    # Now try to instantiate the dataset
    try:
        dataset = Triplets(dummy_emb_layer, dummy_tokeniser)
        print(f"Dataset length: {len(dataset)}")
        qry, pos, neg = dataset[0]
        print("Sample triplet loaded!")
        print(f"Query shape: {qry.shape}, device: {qry.device}")
        print(f"Positive shape: {pos.shape}, device: {pos.device}")
        print(f"Negative shape: {neg.shape}, device: {neg.device}")
    except Exception as e:
        print("Error during dataset test:", e)