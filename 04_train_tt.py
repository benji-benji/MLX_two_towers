import torch
import models
import pickle
import dataset
import datetime
import wandb
import tqdm
from load_glove import load_glove
import torch.nn as nn 

torch.manual_seed(42)
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

with open('./corpus/tokeniser.pkl', 'rb') as f: tkns = pickle.load(f)
words_to_idx, ids_to_words = tkns['words_to_idx'], tkns['ids_to_words']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMBED_DIM = 100
glove_path = '/Users/benjipro/MLX/MLX_two_towers/glove_embeddings/glove.6B.100d.word2vec.embeddings.txt'
embeddings = load_glove(glove_path, words_to_idx, embedding_dim=EMBED_DIM)
embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True)

ds = dataset.Triplets(embedding_layer, words_to_idx)
print("Number of triplets:", len(ds))
dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

two = models.Towers(glove_dim=EMBED_DIM).to(dev)
torch.save(two.state_dict(), f'./checkpoints/{ts}.0.0.two.pth')
print('two:', sum(p.numel() for p in two.parameters())) # 66,048
opt = torch.optim.Adam(two.parameters(), lr=0.003)

wandb.init(project='mlx6-week-02-two',
           config={
             "num_epochs": 1,
             "batch_size": 256,
             "learning_rate": 0.003,
             "margin": 0.3,
             "embedding_dim": 100,
           })

for epoch in range(3):
  prgs = tqdm.tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)
  for idx, (qry, pos, neg) in enumerate(prgs):
    qry, pos, neg = qry.to(dev), pos.to(dev), neg.to(dev)
    loss = two(qry, pos, neg, mrg=0.3)
    opt.zero_grad()
    loss.backward()
    opt.step()
    wandb.log({'loss': loss.item()})
    if idx % 50 == 0: torch.save(two.state_dict(), f'./checkpoints/{ts}.{epoch}.{idx}.two.pth')

#
wandb.finish()