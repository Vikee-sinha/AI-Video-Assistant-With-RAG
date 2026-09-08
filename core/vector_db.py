import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcripts"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )


def get_vector_store(transcript: str) -> Chroma:
    print("Initializing vector store...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(transcript)

    documents = [
        Document(page_content=chunk)
        for chunk in chunks
    ]

    return Chroma.from_documents(
        documents=documents,
        embedding=get_embedding_model(),
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )


def load_vector_store() -> Chroma:
    print("Loading vector store...")

    embeddings = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )


def get_retriever(vector_store: Chroma):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )