import datasets
import hashlib
import pickle
import random

random.seed(42)
ds = datasets.load_dataset('microsoft/ms_marco', 'v1.1')

# Reproduce your eval/train split exactly
complete_qrs = [e['query'] for s in ds.keys() for e in ds[s]]
eval_qrs = set(random.sample(complete_qrs, 50))  # use set for fast lookup
all_qrs = set(q for q in complete_qrs if q not in eval_qrs)

'''

Module to create two dictionaries 

Dictionary 1: 

qrys = {
      query_id_1: {
          'text': 'What is machine learning?',
          'docs': ['abc123def456', 'fed789cba012', ...]  # list of relevant doc hashes
      },
      query_id_2: {
          'text': 'How do computers work?', 
          'docs': ['xyz789abc123', 'def456ghi789', ...]
      },
      # ... more queries
}

Dictionary 2: 

docs = {
    'abc123def456': 'Machine learning is a subset of AI that...',  # hash: actual passage text
    'fed789cba012': 'Neural networks are computational models...',
    'xyz789abc123': 'Computers process information using...',
    # ... more documents
}

'''


docs = {}
qrys = {}
evals = {}

for s in ds.keys():
    for e in ds[s]:
        q_text = e['query']
        q_id = e['query_id']
        # Decide which dict to use
        if q_text in eval_qrs:
            dct = evals
        else:
            dct = qrys
        # Add query entry
        dct[q_id] = { 'text': q_text, 'docs': [] }
        # Add all passages for this query
        for p in e['passages']['passage_text']:
            hsh = hashlib.sha256(p.encode()).hexdigest()[:16]
            if hsh not in docs:
                docs[hsh] = p
            dct[q_id]['docs'].append(hsh)




print("len(qrys)", len(qrys))
print("len(docs)", len(docs))
print(list(docs.keys())[:10])
print(list(qrys.keys())[:10])


#
#
#
print(docs['fdb37125d43984c2'])
print(random.choice(list(docs.values())))
print(qrys[9655])


with open('./corpus/docs.pkl', 'wb') as f: pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('./corpus/qrys.pkl', 'wb') as f: pickle.dump(qrys, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('./corpus/evals.pkl', 'wb') as f: pickle.dump(evals, f, protocol=pickle.HIGHEST_PROTOCOL)
