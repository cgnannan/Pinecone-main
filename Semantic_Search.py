import openai
import os
import pinecone
from datasets import load_dataset
from tqdm.auto import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"

"""
res=openai.Embedding.create(
    input = [
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ],
     model = EMBEDDING_MODEL
)
#print(f"vector 0 :{len(res['data'][0]['embedding'])}\nvector 1 :{len(res['data'][1]['embedding'])}")

# we can extract embeddings to a list
embeds = [record['embedding'] for record in res['data']]
"""

index_name = 'semantic-search-openai'

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key="b4f05738-8211-4414-a372-d0867ef33c10",
    environment="northamerica-northeast1-gcp"  # find next to api key in console
)

"""
# check if 'openai' index already exists (only create index if not)
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=len(embeds[0]))
"""

# connect to index
index = pinecone.Index(index_name)

query = "Why was there a long-term economic downturn in the early 20th century?"
xq=openai.Embedding.create(input=query,model=EMBEDDING_MODEL)['data'][0]['embedding']
res=index.query([xq],top_k=5,include_metadata=True)

for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")

"""
# load the first 1K rows of the TREC dataset
trec = load_dataset('trec', split='train[:1000]')
print(trec[0])

count = 0  # we'll use the count to create unique IDs
batch_size = 32  # process everything in batches of 32
for i in tqdm(range(0, len(trec['text']), batch_size)):
    # set end position of batch
    i_end = min(i+batch_size, len(trec['text']))
    # get batch of lines and IDs
    lines_batch = trec['text'][i: i+batch_size]
    ids_batch = [str(n) for n in range(i, i_end)]
    # create embeddings
    res = openai.Embedding.create(input=lines_batch, model=EMBEDDING_MODEL)
    embeds = [record['embedding'] for record in res['data']]
    # prep metadata and upsert batch
    meta = [{'text': line} for line in lines_batch]
    to_upsert = zip(ids_batch, embeds, meta)
    # upsert to Pinecone
    index.upsert(vectors=list(to_upsert))
"""