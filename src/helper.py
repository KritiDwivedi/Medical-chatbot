from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document


def load_pdf_files(data):
    loader=DirectoryLoader( #directory loader finds the files in the specified directory
        data,
        glob="*.pdf", #loads all pdf files in the directory
        loader_cls=PyPDFLoader #specifies the loader class for PDF files( different loader class for txt,md files)
    )
    documents=loader.load() #loader is the object that loads the documents from the specified directory
    return documents


def filter_documents(documents: List[Document]) -> List[Document]:
    filtered_docs = []
    for doc in documents:
        filtered_doc = Document(
            page_content=doc.page_content,
            metadata={
                "source": doc.metadata.get("source", ""),
            }
        )
        filtered_docs.append(filtered_doc)
    return filtered_docs

# chunking the documents into smaller pieces for better processing  
def text_splitter(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=20, #500 tokens is the size of one chunk
        length_function=len
    )
    texts_chunks = text_splitter.split_documents(documents)
    return texts_chunks

# creating embeddings
from langchain_huggingface import HuggingFaceEmbeddings

def download_embeddings():
    """Download and return the Hugging Face embedding model."""
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name

    )

    return embeddings
embeddings=download_embeddings()