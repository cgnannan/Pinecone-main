import openai
import os
import pinecone



openai.api_key=os.getenv("OPENAI_API_KEY")
model="gpt-4-0613"
embed_model="text-embedding-ada-002"
#data = load_dataset('jamescalam/youtube-transcriptions', split='train')
old_index_name = 'openai'
new_index_name = 'fifa-world-cup-2022-qatar'
query = "Brazil's final place in Qatar 2022 World Cup?"

res=openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ],
    model=embed_model
)

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key="b4f05738-8211-4414-a372-d0867ef33c10",
    environment="northamerica-northeast1-gcp"  # may be different, check at app.pinecone.io
)

"""# Check whether the index with the same name already exists - if so, delete it
if old_index_name in pinecone.list_indexes():
    pinecone.delete_index(old_index_name)

# Creates new index
if new_index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        new_index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='cosine',
        metadata_config={'indexed': ['channel_id', 'published']}
    )
# connect to index"""
index = pinecone.Index(new_index_name)
# view index stats
index_stats=index.describe_index_stats()
print(index_stats)

"""
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

limit = 3750

def retrieve(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=5, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]
    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
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
    return prompt
    
query_with_contexts=retrieve(query)
print(query_with_contexts)

def complete(prompt):
    res=openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=256
    )
    return res['choices'][0]['message']['content'].strip()

print(complete(query_with_contexts))