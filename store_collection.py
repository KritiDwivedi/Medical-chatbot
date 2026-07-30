# just use the existing collection and no need to chunk and embed again and again
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2")


client = chromadb.PersistentClient(
    path="./chroma_db"
)

chroma_collection = client.get_collection(
    name="medical_chatbot",
    embedding_function=embedding_function
)