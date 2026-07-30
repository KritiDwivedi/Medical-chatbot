# creating this file so that if a new pdf loads then only we will have to run this file,so that the collection is updated
# else we can just run the store_collection.py file which will simply use the existing database or collection and hence no need to do chunking,embedding again and again
from dotenv import load_dotenv
import os
from src.helper import load_pdf_files,filter_documents,text_splitter,download_embeddings
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction



load_dotenv()
GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY")



extracted_data=load_pdf_files(data='data/')
print("extracted the data from pdf...")
filter_data=filter_documents(extracted_data)
print("created the chunks...")
text_chunks=text_splitter(filter_data)

embeddings=download_embeddings()

# storing the embeddings in a vector database

# create a new ChromaDB collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "medical_chatbot" 
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)
print("storing the embeddings into a collection...")
ids = []
documents = []
metadatas = []

for i, chunk in enumerate(text_chunks):
    ids.append(f"chunk_{i}")
    documents.append(chunk.page_content)
    metadatas.append(chunk.metadata)


# storing the collection in batches because otherwise the limit of number of records that can be inserted is getting exceeded
BATCH_SIZE = 1000

for i in range(0, len(ids), BATCH_SIZE):
    chroma_collection.add(
        ids=ids[i:i + BATCH_SIZE],
        documents=documents[i:i + BATCH_SIZE],
        metadatas=metadatas[i:i + BATCH_SIZE]
    )

    print(
        f"Indexed {min(i + BATCH_SIZE, len(ids))}/{len(ids)} chunks..."
    )

print("Indexing completed!")

print(f"Successfully stored {len(text_chunks)} chunks in ChromaDB.")

# QUERY THE COLLECTION
# question = "What are the symptoms of diabetes?"
# results=chroma_collection.query(
#     query_texts=[question],
#     n_results=3
# )
# print("Query Results:")

# print(results)