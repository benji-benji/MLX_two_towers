import torch
import pickle
import torch.nn as nn
import dataset
import models
import datetime
import wandb
import tqdm
from load_glove import load_glove

#Hyperparameters
EMBED_DIM = 100 
Batchsize = 32
Margin = 0.1
lr = 0.001
Epochs = 10

wandb.init(project='mlx6-week-02-two',
           config={
             "num_epochs": Epochs,
             "batch_size": Batchsize,
             "learning_rate": lr,
             "margin": Margin,
             "embedding_dim": EMBED_DIM,
           })

# Load tokenizer
with open('./corpus/tokeniser.pkl', 'rb') as f:
    tkns = pickle.load(f)
words_to_idx, ids_to_words = tkns['words_to_idx'], tkns['ids_to_words']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load GloVe embeddings
glove_path = '/Users/aparna/Documents/CollabWeek2/MLX_two_towers/corpus/glove.6B.100d.txt'
embeddings = load_glove(glove_path, words_to_idx, embedding_dim=EMBED_DIM)
embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True)

# Dataset and DataLoader
ds = dataset.Triplets(embedding_layer, words_to_idx)
dl = torch.utils.data.DataLoader(ds, Batchsize, shuffle=True)

# Towers model
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
two = models.Towers(glove_dim=EMBED_DIM).to(dev)
torch.save(two.state_dict(), f'./checkpoints/{ts}.0.0.two.pth')
print('two:', sum(p.numel() for p in two.parameters()))
opt = torch.optim.Adam(two.parameters(), lr=lr)

for epoch in range(Epochs):
    prgs = tqdm.tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)
    for idx, (qry, pos, neg) in enumerate(prgs):
        qry, pos, neg = qry.to(dev), pos.to(dev), neg.to(dev)
        loss = two(qry, pos, neg, mrg=Margin)
        opt.zero_grad()
        loss.backward()
        opt.step()
        wandb.log({'loss': loss.item()})
        if idx % 50 == 0:
            torch.save(two.state_dict(), f'./checkpoints/{ts}.{epoch}.{idx}.two.pth')

torch.save(two.state_dict(), f'./checkpoints/final.two.pth')

wandb.finish()



