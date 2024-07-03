import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex



db = chromadb.PersistentClient(path="chroma_db")
chroma_collection = db.get_or_create_collection('settegiorni')

embed_model = OpenAIEmbedding(model="text-embedding-3-large")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

doc_to_update = chroma_collection.get(limit=1)

# Print the original document
print("Original document:", doc_to_update)
