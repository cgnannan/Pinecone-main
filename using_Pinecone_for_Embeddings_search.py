import openai
from typing import Iterator, List,Iterable
import pandas as pd
import numpy as np
import os
import wget
from ast import literal_eval

#Pinecone Client Library for Python
import pinecone
api_key = os.getenv("PINECONE_API_KEY")
enviroment=os.getenv("PINECONE_ENV")
pinecone.init(api_key=api_key,environment=enviroment,verify=False)

# I've set this to our new embeddings model,this can be changed to the embeddings model of your choice
EMBEDDING_MODEL="text-embeddings-ada-002"

# Ignore unclosed SSL socket warnings - optional in case you get these errors
import warnings

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
#Load data
embeddings_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'
# The file is ~700 MB so this will take some time
wget.download(embeddings_url)

import zipfile
with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip","r") as zip_ref:
    zip_ref.extractall("../data")
"""
article_df = pd.read_csv('../data/vector_database_wikipedia_articles_embedded.csv')

# Read vectors from strings back into a list
article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
article_df['content_vector'] = article_df.content_vector.apply(literal_eval)

# Set vector_id to be a string
article_df['vector_id'] = article_df['vector_id'].apply(str)

# Models a simple batch generator that make chunks out of an input DataFrame
class BatchGenerator:
    
    
    def __init__(self, batch_size: int = 10) -> None:
        self.batch_size = batch_size
    
    # Makes chunks out of an input DataFrame
    def to_batches(self, df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        splits = self.splits_num(df.shape[0])
        if splits <= 1:
            yield df
        else:
            for chunk in np.array_split(df, splits):
                yield chunk

    # Determines how many chunks DataFrame contains
    def splits_num(self, elements: int) -> int:
        return round(elements / self.batch_size)
    
    __call__ = to_batches

df_batcher = BatchGenerator(300)

# Pick a name for the new index
index_name = 'wikipedia-articles'

# Check whether the index with the same name already exists - if so, delete it
if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

# Creates new index
pinecone.create_index(name=index_name, dimension=len(article_df['content_vector'][0]))
index = pinecone.Index(index_name=index_name)

# Confirm our index was created
pinecone.list_indexes()

# Upsert content vectors in content namespace - this can take a few minutes
print("Uploading vectors to content namespace..")
for batch_df in df_batcher(article_df):
    index.upsert(vectors=zip(batch_df.vector_id, batch_df.content_vector), namespace='content')

# Upsert title vectors in title namespace - this can also take a few minutes
print("Uploading vectors to title namespace..")
for batch_df in df_batcher(article_df):
    index.upsert(vectors=zip(batch_df.vector_id, batch_df.title_vector), namespace='title')

# Check index size for each namespace to confirm all of our docs have loaded
index_stats=index.describe_index_stats()
print(index_stats)