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

# Inject custom CSS for UI styling
st.markdown("""
<style>
    body {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: 0.3s;
    }
    .card:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }
    .primary-btn {
        background: linear-gradient(90deg, #2563eb, #1e40af);
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    .primary-btn:hover {
        background: linear-gradient(90deg, #1e40af, #1d4ed8);
    }
    .response-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    .source-item {
        border-left: 3px solid #2563eb;
        padding-left: 10px;
        margin-bottom: 10px;
    }
    .source-item:last-child {
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

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
        os.makedirs("sample_docs", exist_ok=True)
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
            st.markdown(f"<div class='response-box'>{response.choices[0].message.content}</div>", unsafe_allow_html=True)

