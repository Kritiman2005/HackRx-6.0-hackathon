import os
import traceback
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from groq import Groq

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Load environment variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in Render Environment Variables.")
else:
    client = Groq(api_key=GROQ_API_KEY)




# --- Prompt Template ---
prompt = ChatPromptTemplate.from_template("""
You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers. Say 'I don't know' if unsure.

CONTEXT: {context}
QUESTION: {question}

Answer:
""")
output_parser = StrOutputParser()

# --- Groq LLM Call ---
def generate_response(prompt_text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about insurance policy documents."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"❌ Failed to generate response: {e}")
        st.error(traceback.format_exc())
        return ""

# --- Page Config ---
st.set_page_config(page_title="LLM Policy QA", layout="wide")

# --- Custom CSS for Tailwind-like styling ---
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

# ✅ Force CPU for SentenceTransformer
embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# --- Tabs for Modes ---
upload_tab, ask_tab = st.tabs(["📤 Upload Documents", "❓ Ask Question"])

# --- Upload Tab ---
with upload_tab:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📤 Upload Documents")
    st.write("Upload your insurance policy documents (PDF, DOCX, TXT) to start.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt'],
        help="Max file size: 10MB"
    )

    if uploaded_file:
        save_path = os.path.join("sample_docs", uploaded_file.name)
        os.makedirs("sample_docs", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"✅ File saved to {save_path}")

        with st.spinner("🔍 Ingesting file..."):
            from ingest import ingest_files
            ingest_files([save_path])
        st.success("✅ Ingestion complete. Go to 'Ask Question' tab.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Ask Question Tab ---
with ask_tab:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("❓ Ask a Question")
    st.write("Ask anything about your uploaded insurance documents.")

    try:
        vectorstore = FAISS.load_local(
            "./faiss_index", embeddings, allow_dangerous_deserialization=True
        )
    except Exception:
        st.warning("⚠️ No vector store found. Please upload and ingest files first.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    retriever = vectorstore.as_retriever()
    user_query = st.text_area(
        "💬 Your Question",
        placeholder="e.g. What is the coverage for natural disasters?",
        height=100
    )

    if st.button("🚀 Ask Question", key="ask", use_container_width=True):
        if user_query.strip():
            context_docs = retriever.get_relevant_documents(user_query)
            context = "\n\n".join(doc.page_content for doc in context_docs)[:3000]
            final_prompt = prompt.format(context=context, question=user_query)

            with st.spinner("🤖 Generating answer..."):
                answer = generate_response(final_prompt)

            st.markdown("<div class='response-box'>", unsafe_allow_html=True)
            st.subheader("✅ AI Answer")
            st.markdown(answer)
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("📄 Sources"):
                for doc in context_docs:
                    st.markdown(f"<div class='source-item'>*{doc.metadata.get('source', 'Unknown Source')}*</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a question before clicking 'Ask Question'.")
    st.markdown("</div>", unsafe_allow_html=True)