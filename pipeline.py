from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import load_indices_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
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
chroma_collection = db.get_or_create_collection('settegiorni_pipeline')

# define embedding function
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# set up ChromaVectorStore and save data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
        TitleExtractor(),
        OpenAIEmbedding(model="text-embedding-3-large")
    ],
    vector_store=vector_store
)

docs = SimpleDirectoryReader(input_dir='./Settegiorni/pdf').load_data()

for doc in docs:
    doc.text = doc.text.replace('\n', ' ').replace('- ', '')

# inserisce direttamente i nodi dentro al db vettoriale

pipeline.run(documents=docs)




# crea l'indice

index = VectorStoreIndex.from_vector_store(
    vector_store
)


