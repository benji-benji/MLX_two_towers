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
