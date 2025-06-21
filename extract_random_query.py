import random
import pickle

evals_path = "/Users/benjipro/MLX/MLX_two_towers/corpus/evals.pkl"

with open(evals_path, 'rb') as f:
        eval_data = pickle.load(f)  # List of (query, correct_doc) pairs
        
print (random.sample(list(eval_data.items()), 5))
