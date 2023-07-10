import openai
import os
import pandas as pd
import pinecone
import ast
from tqdm.auto import tqdm
import time

start_time = time.time()

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = "b4f05738-8211-4414-a372-d0867ef33c10"
pinecone_env = "northamerica-northeast1-gcp" 

model="gpt-3.5-turbo-0613"
embed_model="text-embedding-ada-002"

#data = load_dataset('jamescalam/youtube-transcriptions', split='train') 
csv_file_path = "/Volumes/work/Project/AIGC/OpenAI/Function_Call/data/FIFA_World_Cup_2022.csv"

pinecone_old_index_name = 'fifa-world-cup-2022-qatar'
pinecone_new_index_name = 'qatar-2022-fifa-world-cup'


"""def pinecone_csv_upsert(csv_file=None,old_index_name=None,new_index_name=None):
    "Iterate over the CSV data and upload vectors in a loop"
    csv_file=csv_file_path
    csv_data=pd.read_csv(csv_file) #import data from the csv file
    old_index_name=pinecone_old_index_name
    new_index_name=pinecone_new_index_name

    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key = pinecone_api_key,
        environment = pinecone_env   # may be different, check at app.pinecone.io
    )

    res=openai.Embedding.create(
        input=[
            "Sample document text goes here",
            "there will be several phrases in each batch"
        ],
        model="text-embedding-ada-002"
    )

    # Check whether the index with the same name already exists - if so, delete it
    if old_index_name in pinecone.list_indexes():
        pinecone.delete_index(old_index_name)

    # Creates new index
    if new_index_name not in pinecone.list_indexes():
        # if does not exist, create index
        pinecone.create_index(
            new_index_name,
            dimension=len(res['data'][0]['embedding']),
            metric='cosine'
        )
    # connect to index
    index=pinecone.Index(new_index_name)

    for idx in tqdm(range(len(csv_data)), desc="Uploading vectors"):
        text = csv_data.loc[idx, "text"]
        embedding = ast.literal_eval(csv_data.loc[idx, "embedding"])
        
        # Upsert vector and text to Pinecone index
        index.upsert([(str(idx), embedding, {'text': text})])
    return print(f"text and embedding have been successfully inserted into Pinecone index\nIndex_Name:{new_index_name}\n{index.describe_index_stats()}")

pinecone_csv_upsert()"""

"""
# Iterate over the non-CSV data and upload vectors in a loop
new_data = []
window = 20  # number of sentences to combine
stride = 4  # number of sentences to 'stride' over, used to create overlap

for i in tqdm(range(0, len(data), stride)):
    i_end = min(len(data)-1, i+window)
    if data[i]['title'] != data[i_end]['title']:
        # in this case we skip this entry as we have start/end of two videos
        continue
    text = ' '.join(data[i:i_end]['text'])
    # create the new merged dataset
    new_data.append({
        'start': data[i]['start'],
        'end': data[i_end]['end'],
        'title': data[i]['title'],
        'text': text,
        'id': data[i]['id'],
        'url': data[i]['url'],
        'published': data[i]['published'],
        'channel_id': data[i]['channel_id']
    })

batch_size = 100  # how many embeddings we create and insert at once

for i in tqdm(range(0, len(new_data), batch_size)):
    # find end of batch
    i_end = min(len(new_data), i+batch_size)
    meta_batch = new_data[i:i_end]
    # get ids
    ids_batch = [x['id'] for x in meta_batch]
    # get texts to encode
    texts = [x['text'] for x in meta_batch]
    # create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                done = True
            except:
                pass
    embeds = [record['embedding'] for record in res['data']]
    # cleanup metadata
    meta_batch = [{
        'start': x['start'],
        'end': x['end'],
        'title': x['title'],
        'text': x['text'],
        'url': x['url'],
        'published': x['published'],
        'channel_id': x['channel_id']
    } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)"""


# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key = pinecone_api_key,
    environment = pinecone_env   # may be different, check at app.pinecone.io
)

limit = 3750

# connect to index
index = pinecone.Index(pinecone_new_index_name)
query = "Who is the top scorer of the 2022 Qatar World Cup?"

def ask_pinecone(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=100, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]
    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    response=openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=256
    )
    return response['choices'][0]['message']['content'].strip()

print(ask_pinecone(query))

end_time=time.time()
execution_time = end_time - start_time
print(execution_time)