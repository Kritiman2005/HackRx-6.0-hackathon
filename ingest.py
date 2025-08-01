# ingest.py

from document_loader import load_documents
from vector_store import create_vector_store

def ingest_files(paths):
    documents = load_documents(paths)
    create_vector_store(documents)