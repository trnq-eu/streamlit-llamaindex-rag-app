import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.schema import MetadataMode
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
import openai
import chromadb
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_key = st.secrets.openai_key

# carica i dati direttamente dal database di chromadb



st.header("Interroga una base di conoscenza personalizzata")

if "messages" not in st.session_state.keys(): # Inizializza lo storico dei messaggi del chatbot
    st.session_state.messages = [
        {"role": "assistant", "content": "Sono qui per aiutarti ad analizzare una digital library. Fammi una domanda e cercherò la risposta all'interno dei testi a mia disposizione, citando la fonte delle informazioni."}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    # https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index/
    with st.spinner('Sto caricando e indicizzando i documenti. Potrebbe volerci qualche minuto'):
        # carica i dati dal database di Chroma
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection('settegiorni')
        embed_model = OpenAIEmbedding(model="text-embedding-3-large")

        # assign chroma as the vector_store to the context
        vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
        index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
        )

        # Create an instance of Settings with the OpenAI configuration
        # Configure the Settings class with the OpenAI instance
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
        return index
    
index = load_data()

chat_engine = index.as_chat_engine(chat_mode="context", 
                                   system_prompt=(
        '''Sei un archivista professionista. Rispondi alle domande che ti vengono fatte
        cercando le informazioni all'interno della rivista Settegiorni.
        Privilegia le informazioni reperite all'interno della rivista Settegiorni rispetto alla tua conoscenza pregressa
        Cita sempre il numero della rivista e l'articolo da cui hai preso le informazioni.'''
    ),
                                   verbose=True)

if prompt := st.chat_input('La tua domanda'): # Chiede all'utente di fare una domanda
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Mostra il messaggio precedente
    with st.chat_message(message["role"]):
        st.write(message["content"])
        

# Se l'ultimo messaggio non è dell'assistant, genera una nuova risposta
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Sto riflettendo..."):
            response = chat_engine.chat(prompt)
            # risposa con metadati
            # st.write(response.response, response.source_nodes[0].metadata)
            st.write(response.response)
            # risposta che contiene solo la stringa di risposta
            # st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history

