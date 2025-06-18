
import pandas as pd
# DUMMY DATA

queries = [
    "what is rba",
    "who wrote cat in the hat"
]

docs = [ 
    " results based accountability is a disciplined way",
    " the cat in the hat is a childrens book by dr seuss" 
]

triples = [ 
(0,0,1), # query 0, positive doc 0, negative doc 1
(1,1,0), # query 1, positive doc 1, negative doc 0
]

# MINI REAL DATA

def read_parquet_file(file_path):
    """
    Reads a Parquet file and returns a DataFrame.
    
    Args:
        file_path (str): The path to the Parquet file.
        
    Returns:
        pd.DataFrame: The DataFrame containing the data from the Parquet file.
    """
    return pd.read_parquet(file_path)

dataset_pd = read_parquet_file("train-00000-of-00001.parquet")

def create_passage_query_df(dataset):
    """
    Create a DataFrame with 'passage_text' and 'query' columns.
    
    Args:
        dataset (pd.DataFrame): The input DataFrame containing passages and queries.
        
    Returns:
        pd.DataFrame: A new DataFrame with 'passage_text' and 'query' columns.
    """
    data = []
    for _, row in dataset.iterrows():
        passage_texts = row['passages'].get('passage_text', [])
        query_str = row['query']

        for line in passage_texts:
            data.append({'passage_text': line.strip(), 'query_str': query_str})

    
    return pd.DataFrame(data)

passage_query_df = create_passage_query_df(dataset_pd)

print(type(passage_query_df))
print(passage_query_df.shape)
print(passage_query_df.columns)

mini_sample_df = passage_query_df.sample(n=10, random_state=42)

mini_real_queries_list = mini_sample_df['query_str'].tolist()
mini_real_passages_list = mini_sample_df['passage_text'].tolist()

print("3 random queries:\n")

for i, x in enumerate(mini_real_queries_list):
    if i in (2,5,6):
        print(x)
print("\n")

print("3 corresponding docs:\n")

for i, x in enumerate(mini_real_passages_list):
    if i in (2,5,6):
        print(x)