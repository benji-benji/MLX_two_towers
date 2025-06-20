import datasets 
import requests 
import os

full_dataset = datasets.load_dataset('microsoft/ms_marco','v1.1')
train_dataset = datasets.load_dataset('microsoft/ms_marco','v1.1', split="train")
validation_dataset = datasets.load_dataset('microsoft/ms_marco','v1.1', split="validation")
test_dataset = datasets.load_dataset('microsoft/ms_marco','v1.1', split="test")                              


all_docs = [passage for s in full_dataset.keys() for e in full_dataset[s] for passage in e['passages']['passage_text']]
all_qrs = [e['query'] for s in full_dataset.keys()for e in full_dataset[s]]
train_docs = [passage for s in train_dataset.keys() for e in train_dataset[s] for passage in e['passages']['passage_text']]
train_qrs = [e['query'] for s in train_dataset.keys()for e in train_dataset[s]]
val_docs = [passage for s in validation_dataset.keys() for e in validation_dataset[s] for passage in e['passages']['passage_text']]
val_qrs = [e['query'] for s in validation_dataset.keys()for e in validation_dataset[s]]
test_docs = [passage for s in test_dataset.keys() for e in test_dataset[s] for passage in e['passages']['passage_text']]
test_qrs = [e['query'] for s in test_dataset.keys()for e in test_dataset[s]]

# SENSE CHECK PRINTOUTS 

print("## Documents loaded ##")
print("\n")
print("example doc:\n"+(all_docs[1]))
print("number of docs:"+str(len(all_docs)))
print("\n")    
print("example query:\n"+(all_qrs[1]))
print("number of queries:"+str(len(all_qrs)))                                  

# COMBINE & SAVE TXT FILES

with open('./corpus/msmarco.txt', 'w', encoding='utf-8') as f: f.write('\n'.join(set(all_docs +all_qrs)))
with open('./corpus/msmarco_train.txt', 'w', encoding='utf-8') as f: f.write('\n'.join(set(train_docs +train_qrs)))
with open('./corpus/msmarco_validation.txt', 'w', encoding='utf-8') as f: f.write('\n'.join(set(val_docs +val_qrs)))
with open('./corpus/msmarco_test.txt', 'w', encoding='utf-8') as f: f.write('\n'.join(set(test_docs +test_qrs)))

r = requests.get('https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8')
with open('./corpus/text8.txt', 'wb') as f: f.write(r.content)

# CHECK FILES EXIST
print("\nChecking files saved...\n")  
# msmarco_path = "/Users/aparna/Documents/CollabWeek2/MLX_two_towers/corpus/msmarco.txt"
# text8_path = "/Users/aparna/Documents/CollabWeek2/MLX_two_towers/corpus/text8.txt"
# print ("msmarco.txt file exists:", os.path.exists(msmarco_path))
# print ("text8.txt file exists:", os.path.exists(text8_path))


