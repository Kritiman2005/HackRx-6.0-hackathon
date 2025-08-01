import os
import time
import traceback
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from document_loader import load_documents
from vector_store import create_vector_store

# --- Load environment variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ Please set GROQ_API_KEY in your .env file.")
    st.stop()

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
        print("🧠 Prompt sent to Groq:\n", prompt_text)

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

# --- Streamlit App ---
st.set_page_config(page_title="LLM Policy QA", layout="centered")
st.title("🧠 Chat with Policy Documents")

choice = st.sidebar.selectbox("Choose Mode", ['Upload', 'Ask Question'])

# ✅ FIX: Avoid meta tensor error by setting device to CPU
embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

if choice == 'Upload':
    uploaded_file = st.file_uploader("📎 Upload a PDF, DOCX or TXT", type=['pdf', 'docx', 'txt'])
    if uploaded_file:
        save_path = os.path.join("sample_docs", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"✅ File saved to {save_path}")
        with st.spinner("🔍 Ingesting file..."):
            from ingest import ingest_files
            ingest_files([save_path])
        st.success("✅ Ingestion complete. You can now switch to 'Ask Question'.")

else:
    try:
        vectorstore = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception:
        st.warning("⚠️ Could not load existing vector store. Please upload and ingest files first.")
        st.stop()

    retriever = vectorstore.as_retriever()
    user_query = st.text_input("🔎 Enter your question:")

    if user_query.strip():
        context_docs = retriever.get_relevant_documents(user_query)
        context = "\n\n".join(doc.page_content for doc in context_docs)
        context = context[:3000]  # Truncate if needed

        final_prompt = prompt.format(context=context, question=user_query)

        with st.spinner("🤖 Generating answer..."):
            response = generate_response(final_prompt)
            st.markdown(response)

            with st.expander("📄 Sources"):
                for doc in context_docs:
                    st.markdown(f"- {doc.metadata.get('source', 'Unknown Source')}")


