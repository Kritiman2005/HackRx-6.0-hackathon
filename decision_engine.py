import os
from dotenv import load_dotenv
from groq import Groq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("❌ GROQ_API_KEY is not set in the .env file.")

client = Groq(api_key=GROQ_API_KEY)

prompt = ChatPromptTemplate.from_template("""
You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers. Say 'I don't know' if unsure.

CONTEXT: {context}
QUESTION: {question}

Answer:
""")

output_parser = StrOutputParser()

def ask_groq(question: str, context: str) -> str:
    final_prompt = prompt.format(context=context, question=question)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about insurance policies."},
                {"role": "user", "content": final_prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=512,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error generating answer: {e}"


