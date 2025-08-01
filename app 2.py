import os
import streamlit as st
import time
from dotenv import load_dotenv

from huggingface_hub import InferenceClient
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from document_loader import load_documents
from vector_store import load_vector_store
from ingest import ingest_files

# --- Setup ---
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print("🔑 Token:", HF_TOKEN)

if not HF_TOKEN:
    st.error("❌ Please set HUGGINGFACEHUB_API_TOKEN in your .env file.")
    st.stop()

# --- HuggingFace InferenceClient ---
client = InferenceClient(
    model="tngtech/DeepSeek-TNG-R1T2-Chimera",
    token=HF_TOKEN,
)

def generate_response(prompt_text):
    response = client.text_generation(
        prompt=prompt_text,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    )
    return response.strip()

# --- Prompt Template ---
prompt = ChatPromptTemplate.from_template("""
You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers. Say "I don't know" if unsure.

CONTEXT: {context}
QUESTION: {question}

Answer:
""")
output_parser = StrOutputParser()

# --- UI ---
st.set_page_config(page_title="LLM Policy QA", layout="centered")
st.title("🧠 Chat with Policy Documents")

choice = st.sidebar.selectbox("Choose Mode", ['Upload', 'Ask Question'])

if choice == 'Upload':
    uploaded_file = st.file_uploader("Upload a PDF, DOCX or TXT", type=['pdf', 'docx', 'txt'])
    if uploaded_file:
        save_path = os.path.join("sample_docs", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"✅ File saved to {save_path}")
        with st.spinner("🔍 Ingesting file..."):
            ingest_files([save_path])
        st.success("✅ Ingestion complete. You can now switch to 'Ask Question'.")

else:  # Ask Question
    try:
        vectorstore = load_vector_store()
    except Exception:
        st.warning("⚠️ Could not load existing vector store. Ingest documents first.")
        st.stop()

    retriever = vectorstore.as_retriever()
    user_query = st.text_input("🔎 Enter your query:")

    if user_query.strip():
        context_docs = retriever.get_relevant_documents(user_query)
        if not context_docs:
            st.warning("⚠️ No relevant documents found for your query.")
            st.stop()

        context = "\n\n".join(doc.page_content for doc in context_docs)

        chain = (
            {
                "context": lambda x: context,
                "question": RunnablePassthrough()
            }
            | prompt
            | (lambda prompt_input: generate_response(prompt_input))
            | output_parser
        )

        with st.spinner("🤖 Generating answer..."):
            response = chain.invoke(user_query)
            st.markdown(response)

