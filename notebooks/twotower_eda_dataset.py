# open train-00000-of-00001.parquet and read the data
import pandas as pd


def read_parquet_file(file_path):
    """
    Reads a Parquet file and returns a DataFrame.
    
    Args:
        file_path (str): The path to the Parquet file.
        
    Returns:
        pd.DataFrame: The DataFrame containing the data from the Parquet file.
    """
    return pd.read_parquet(file_path)

dataset_pd = read_parquet_file("../data/raw/train-00000-of-00001.parquet")
# Display the shape, first few rows, columns, data types, null values, descriptive statistics, and info of the DataFrame
print(dataset_pd.shape)
print(dataset_pd.head())
print(dataset_pd.columns)
print(dataset_pd.dtypes)
print(dataset_pd.isnull().sum())
print(dataset_pd.describe())
print(dataset_pd.info())

print(dataset_pd.passages[0])  # Display the first row of the passages colum

print(dataset_pd.passages[0].get('passage_text'))  # Display the passage text of the first row
# display row 1, column passage and column query
print("DOCS=")
print(dataset_pd.iloc[0]["passages"].keys())
print("QUERY=")
print(dataset_pd.iloc[0]["query"])

ten_passages = dataset_pd.passages[0].get('passage_text')
print(type(ten_passages))
print(ten_passages)
print(ten_passages.shape)
# Display the shape of ten_passages
print(dataset_pd["query"][0])
query_df = dataset_pd["query"][0]
ten_passages_plus_query = ten_passages + query_df
print(ten_passages_plus_query)



# a function to iterate through all rows of dataset_pd
# and create a new DataFrame with each line of 'passage_text'
# and 'query' columns
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

print(passage_query_df.shape)
print(passage_query_df.columns) 
print(passage_query_df.passage_text.get(0))
print(passage_query_df.passage_text.get(1))
print(passage_query_df.query_str.get(0))
print(passage_query_df.query_str.get(1))

print(passage_query_df.passage_text.get(9))
print(passage_query_df.passage_text.get(11))
print(passage_query_df.query_str.get(9))
print(passage_query_df.query_str.get(11))

print(passage_query_df.head(20))










#passage_query_df = ........

#print(passage_query_df.shape)
#print(passage_query_df.head())
#print(passage_query_df.columns)