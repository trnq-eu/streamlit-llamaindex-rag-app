from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import load_indices_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
import openai
import chromadb
import logging
import sys
import streamlit as st

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_key = st.secrets.openai_key


db = chromadb.PersistentClient(path="./chroma_db")

# create a chroma collection
chroma_collection = db.get_or_create_collection('digitallib')

# define embedding function
embed_model = OpenAIEmbedding(model="text-embedding-3-large")


# load documents
docs = SimpleDirectoryReader(input_dir='./data').load_data()

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    docs, storage_context=storage_context, embed_model=embed_model
)









splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(docs)
        
# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db")

# create a chroma collection
chroma_collection = db.get_or_create_collection('digitallib')

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection = chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(nodes)



