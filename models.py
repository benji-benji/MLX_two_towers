
import torch
import torch.nn as nn
from torch import Tensor

glove_dim = 100

class Query_Tower(torch.nn.Module):
    """
        Args:
            input_dim: Dimension of pre-trained embeddings (GloVe:100)
    """
    
    def __init__(self, glove_dim):
        super().__init__()
        self.linearlayer_1 = torch.nn.Linear(in_features=glove_dim, out_features=glove_dim)
        self.ReLUlayer_1= torch.nn.ReLU()
        self.linearlayer_2= torch.nn.Linear(in_features=glove_dim, out_features=glove_dim)

    def forward(self, average: Tensor):
        out_layer1 = self.linearlayer_1(average)
        out_layer2 = self.ReLUlayer_1(out_layer1)
        out_layer3 = self.linearlayer_2(out_layer2)
        return out_layer3
    
class Doc_Tower(torch.nn.Module):
    """
        Args:
            input_dim: Dimension of pre-trained embeddings (GloVe:100)      
    """
    
    def __init__(self, glove_dim):
        super().__init__()
        self.linearlayer_1 = torch.nn.Linear(in_features=glove_dim, out_features=glove_dim)
        self.ReLUlayer_1= torch.nn.ReLU()
        self.linearlayer_2= torch.nn.Linear(in_features=glove_dim, out_features=glove_dim)

    def forward(self, average: Tensor):
        out_layer1 = self.linearlayer_1(average)
        out_layer2 = self.ReLUlayer_1(out_layer1)
        out_layer3 = self.linearlayer_2(out_layer2)
        return out_layer3

class Towers(torch.nn.Module):
    def __init__(self, glove_dim):
        super().__init__()
        self.query_tower = Query_Tower(glove_dim)
        self.doc_tower = Doc_Tower(glove_dim)

    def forward(self, query, posdoc, negdoc, mrg):
        query_vector = self.query_tower(query)
        posdoc_vector = self.doc_tower(posdoc)
        negdoc_vector = self.doc_tower(negdoc)
        positive_similarity = 1 - nn.functional.cosine_similarity(query_vector, posdoc_vector)
        negative_similarity = 1 - nn.functional.cosine_similarity(query_vector, negdoc_vector)
        return torch.max(positive_similarity - negative_similarity + mrg, torch.tensor(0.0)).mean()














