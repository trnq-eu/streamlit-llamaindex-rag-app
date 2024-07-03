import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex



db = chromadb.PersistentClient(path="chroma_db")
chroma_collection = db.get_or_create_collection('settegiorni_anno_1')



docs = chroma_collection.get()

# if 'documents' in docs and docs['documents']:
#     docs['documents'] = [doc.replace('\n', ' ').replace('- ', '') for doc in docs['documents']]


# Print the original document
print("Original document:", docs)

