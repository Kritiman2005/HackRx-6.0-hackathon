import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set Streamlit config
st.set_page_config(page_title="Groq QA App", layout="wide")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS vector store if exists
if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = None

st.title("🔍 Policy Q&A - Groq + LangChain")

mode = st.sidebar.radio("Select Mode", ["Ask Question", "Upload Docs"])

if mode == "Upload Docs":
    uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True)
    if uploaded_files:
        from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        docs = []
        for file in uploaded_files:
            file_path = os.path.join("sample_docs", file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(file_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue

            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local("faiss_index")
        st.success("✅ Documents uploaded and indexed successfully!")

elif mode == "Ask Question":
    if vectorstore is None:
        st.warning("Please upload and index documents first.")
    else:
        question = st.text_input("Enter your question")
        if question:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(question)

            context = "\n".join([d.page_content for d in docs])

            prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {question}\nAnswer:"
            response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            st.markdown("**Answer:**")
            st.write(response.choices[0].message.content)


