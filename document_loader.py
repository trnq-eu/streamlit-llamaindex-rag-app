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

# questo script carica i dati e crea gli indici all'interno di una collezione di chroma db
# l'utilizzo di questo script è dispendioso: usare con cautela

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_key = st.secrets.openai_key


db = chromadb.PersistentClient(path="./chroma_db")

# create a chroma collection
chroma_collection = db.get_or_create_collection('settegiorni')

# define embedding function
embed_model = OpenAIEmbedding(model="text-embedding-3-large")


# load documents
docs = SimpleDirectoryReader(input_dir='./Settegiorni/pdf').load_data()


# # set up ChromaVectorStore and load in data
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(
#     docs, storage_context=storage_context, embed_model=embed_model
# )

for doc in docs:
    doc.text = doc.text.replace('\n', ' ').replace('- ', '')


        
# set up ChromaVectorStore and save data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    docs, storage_context=storage_context, embed_model=embed_model
)


