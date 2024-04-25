import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

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
        reader = SimpleDirectoryReader(input_dir='./data', recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model='gpt-3.5-turbo', 
                                                                  temperature=0.2, 
                                                                  system_prompt='''Sei un archivista professionista e il tuo ruolo è quello di fare delle ricerche storiche basandoti su delle fonti autorevoli. 
                                                                  Rispondi alle domande citando la fonte delle informazioni che hai raccolto ed esponi i fatti con autorevolezza e precisione. 
                                                                  Non inventare nulla.''')),
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
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

