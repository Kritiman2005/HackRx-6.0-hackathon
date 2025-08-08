# app.py (updated for PDF embedding during upload)
import os
import traceback
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# --- Setup ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# --- Helper to query Groq ---
def ask_groq(prompt):
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful insurance underwriting assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.error(traceback.format_exc())
        return ""

# --- Streamlit UI ---
st.set_page_config(page_title="📑 Application Review Assistant", layout="wide")
st.title("📑 Insurance Application Advisor")

mode = st.sidebar.selectbox("Choose Mode", ["Upload", "Review & Score"])

# Use SentenceTransformer embeddings
embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

INDEX_PATH = "./faiss_index"
DOCS_DIR = "sample_docs"
os.makedirs(DOCS_DIR, exist_ok=True)

if mode == "Upload":
    uploaded_file = st.file_uploader("Upload application (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
    if uploaded_file:
        save_path = os.path.join(DOCS_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"✅ File saved to {save_path}")

        # --- Load document based on type ---
        if uploaded_file.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(save_path)
        elif uploaded_file.name.lower().endswith(".docx"):
            loader = Docx2txtLoader(save_path)
        else:
            loader = TextLoader(save_path)

        docs = loader.load()

        # --- Create or update FAISS index ---
        if os.path.exists(INDEX_PATH):
            vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents(docs)
        else:
            vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(INDEX_PATH)

        st.success("✅ File embedded and added to vector store. Ready for review!")

elif mode == "Review & Score":
    if not os.path.exists(INDEX_PATH):
        st.error("⚠️ No ingested data found. Upload files first.")
        st.stop()

    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    user_query = st.text_input("🔎 Describe what to review (e.g., 'insurance for 46M with surgery in Pune'):")

    if user_query:
        docs = retriever.get_relevant_documents(user_query)
        if not docs:
            st.warning("No relevant documents found.")
            st.stop()

        context = "\n\n".join(doc.page_content for doc in docs)[:3000]

        # --- Generate Score ---
        with st.spinner("📊 Scoring application..."):
            score_prompt = f"""
            Evaluate this insurance application and return a score (0-100) representing the likelihood of acceptance.
            Justify your answer briefly.

            APPLICATION:
            {context}
            """
            score_response = ask_groq(score_prompt)
            st.subheader("📈 Approval Likelihood")
            st.markdown(score_response)

        # --- Generate Suggestions ---
        with st.spinner("🛠 Suggesting improvements..."):
            improvement_prompt = f"""
            Suggest specific ways to improve this insurance application so it has a higher chance of acceptance.
            Focus on coverage, clarity, duration, medical history, or premium options.

            APPLICATION:
            {context}
            """
            improvement_response = ask_groq(improvement_prompt)
            st.subheader("💡 Suggestions for Improvement")
            st.markdown(improvement_response)

        with st.expander("📄 Sources"):
            for doc in docs:
                st.markdown(f"- {doc.metadata.get('source', 'Unknown Source')}")
