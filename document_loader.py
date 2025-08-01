import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def load_documents(file_paths=None):
    docs = []
    if not file_paths:
        if not os.path.exists("sample_docs"):
            raise FileNotFoundError("The 'sample_docs' directory does not exist.")
        file_paths = [os.path.join("sample_docs", f)
                      for f in os.listdir("sample_docs")
                      if not f.startswith(".")]

    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif path.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)




