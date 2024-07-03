import csv
import openai
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

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# questo script server per caricare i dati aggiungendo metadati ai documenti

openai.api_key = st.secrets.openai_key

# Lista per memorizzare i documenti
documents = []

csv_file = './Settegiorni/Settegiorni.csv'

# # Leggi i dati dal file CSV e aggiungi i metadati ai documenti
with open(csv_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Carica il documento e aggiungi metadati
        docs = SimpleDirectoryReader(input_files=[row['Path']]).load_data()
        # preprocessamento del testo per rimuovere gli a capo e "- "
        if docs:
            docs[0].text = docs[0].text.replace('\n', ' ').replace('- ', '')
            docs[0].metadata['url'] = row['url']
            docs[0].metadata['title'] = row['Titoli degli oggetti']
            docs[0].metadata['description'] = row['Descrizione']
            docs[0].metadata['related people'] = row['entità collegate/i']
            documents.append(docs)


# metadata extractor
# https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtraction_LLMSurvey/

db = chromadb.PersistentClient(path="./chroma_db")

# create a chroma collection
chroma_collection = db.get_or_create_collection('settegiorni')

# define embedding function
embed_model = OpenAIEmbedding(model="text-embedding-3-large")


# # load documents
# docs = SimpleDirectoryReader(input_dir='./data').load_data()

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)