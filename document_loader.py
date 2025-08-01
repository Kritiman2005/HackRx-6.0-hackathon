# document_loader.py

import os
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

def load_documents(paths):
    docs = []
    for path in paths:
        if os.path.isdir(path):
            for file in os.listdir(path):
                full_path = os.path.join(path, file)
                docs.extend(load_documents([full_path]))
        else:
            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif path.endswith(".txt"):
                loader = TextLoader(path)
            elif path.endswith(".docx"):
                loader = Docx2txtLoader(path)
            else:
                continue
            docs.extend(loader.load())
    return docs