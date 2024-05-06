import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import openai
import chromadb
# fonte: https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/

openai.api_key = st.secrets.openai_key

st.header("Interroga una base di conoscenza personalizzata")

if "messages" not in st.session_state.keys(): # Inizializza lo storico dei messaggi del chatbot
    st.session_state.messages = [
        {"role": "assistant", "content": "Sono qui per aiutarti ad analizzare una digital library. Fammi una domanda e cercherò la risposta all'interno dei testi a mia disposizione, citando la fonte delle informazioni."}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    # https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index/
    with st.spinner('Sto caricando e indicizzando i documenti. Potrebbe volerci qualche minuto'):
        # reader = SimpleDirectoryReader(input_dir='./data', recursive=True)
        # docs = reader.load_data()
        docs = SimpleDirectoryReader(input_dir='./data').load_data()
        
        # initialize client, setting path to save data
        db = chromadb.PersistentClient(path="./chroma_db")

        # create a chroma collection
        chroma_collection = db.get_or_create_collection('digitallib')

        # assign chroma as the vector_store to the context
        vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)


        # Create an instance of Settings with the OpenAI configuration
        # Configure the Settings class with the OpenAI instance
        Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        index = VectorStoreIndex.from_documents(docs, storage_context = storage_context)
        return index
    
index = load_data()

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

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
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history

