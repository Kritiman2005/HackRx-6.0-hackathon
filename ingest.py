from document_loader import load_documents, split_documents
from vector_store import create_vector_store

def ingest_files(file_paths=None):
    docs = load_documents(file_paths)
    chunks = split_documents(docs)
    create_vector_store(chunks)



