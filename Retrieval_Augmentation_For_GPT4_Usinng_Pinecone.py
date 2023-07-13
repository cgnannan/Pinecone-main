import os
from bs4 import BeautifulSoup
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
from tqdm.auto import tqdm
import openai
import pinecone
import datetime
from time import sleep
from IPython.display import Markdown

openai.api_key = os.getenv("OPENAI_API_KEY")

embed_model="text-embedding-ada-002"
HTML_PATH='rtdocs'

pinecone_old_index_name = 'wikipedia-articles'
pinecone_new_index_name = 'gpt-4-langchain-docs'

res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], 
    model=embed_model
)

# Define the named tuple

"""def extract_docs(root_dir):
    # 存储所有文本的列表
    docs = []

    # 遍历 'rtdocs' 文件夹及其所有子文件夹
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 如果文件是 HTML 文件
            if file.endswith('.html'):
                # 拼接完整的文件路径
                source = os.path.join(root, file)

                # 打开并读取文件
                with open(source, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 使用 BeautifulSoup 解析 HTML
                soup = BeautifulSoup(content, 'html.parser')

                # 提取文本并添加到列表中
                page_content = soup.get_text()

                # 将路径和页面内容作为元组添加到列表中
                docs.append(
                    {"source" : f"{source}",
                    "page_content": f"{page_content}"}
                )
    return docs

docs=extract_docs(HTML_PATH)

data = []

for doc in docs:
    data.append({
        'url': doc['source'].replace('rtdocs/', 'https://'),
        'text': doc['page_content']
    })

tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = []

for idx, record in enumerate(tqdm(data)):
    texts = text_splitter.split_text(record['text'])
    chunks.extend([{
        'id': str(uuid4()),
        'text': texts[i],
        'chunk': i,
        'url': record['url']
    } for i in range(len(texts))])"""

# initialize connection to pinecone
pinecone.init(
    api_key = "e2a1bdfc-46b8-4045-b05d-0cbcecb0cbb8",  # app.pinecone.io (console)
    environment = "us-west4-gcp-free"  # next to API key in console
)

# Check whether the index with the same name already exists - if so, delete it
if pinecone_old_index_name in pinecone.list_indexes():
    pinecone.delete_index(pinecone_old_index_name)

# Creates new index
if pinecone_new_index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        pinecone_new_index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='cosine'
    )

# connect to index
index = pinecone.GRPCIndex(pinecone_new_index_name)
# view index stats
print(index.describe_index_stats())

"""batch_size = 10  # how many embeddings we create and insert at once

for i in tqdm(range(0, len(chunks), batch_size)):
    # find end of batch
    i_end = min(len(chunks), i+batch_size)
    meta_batch = chunks[i:i_end]
    # get ids
    ids_batch = [x['id'] for x in meta_batch]
    # get texts to encode
    texts = [x['text'] for x in meta_batch]
    # create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input=texts, model=embed_model)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, model=embed_model)
                done = True
            except:
                pass
    embeds = [record['embedding'] for record in res['data']]
    # cleanup metadata
    meta_batch = [{
        'text': x['text'],
        'chunk': x['chunk'],
        'url': x['url']
    } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)"""

query = "how do I use the LLMChain in LangChain?"

res = openai.Embedding.create(
    input=query,
    model=embed_model
)

# retrieve from Pinecone
xq = res['data'][0]['embedding']

# get relevant contexts (including the questions)
res = index.query(xq, top_k=5, include_metadata=True)
     
# get list of retrieved text
contexts = [item['metadata']['text'] for item in res['matches']]

augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query

res = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are Q&A bot. A highly intelligent system that answers user questions based on the information provided by the user above each question. If the information can not be found in the information provided by the user you truthfully say 'I don't know'"},
        {"role": "user", "content": augmented_query}
    ]
)
print(res['choices'][0]['message']['content'])