# HackRx-6.0-hackathon
# 🧠 LLM Policy QA & Application Advisor

This repository provides a unified platform powered by LLMs to:

1. **Chat with Policy Documents** – Ask questions from insurance PDFs or DOCX files and receive grounded answers.
2. **Application Assessment Assistant** – Score the approval likelihood of insurance applications.
3. **Query-Based Retrieval** – Extract document insights using structured natural language queries (e.g. "46M, knee surgery, Pune, 3-month policy").

---

## 🚀 Features

### 📄 1. Document Q&A
- Upload PDF, DOCX, or TXT policy files
- Content is embedded via `sentence-transformers/all-MiniLM-L6-v2`
- Uses FAISS for vector search and Groq’s `llama3-70b-8192` for answering
- Natural language queries yield document-grounded answers

**Example Output:**
> "Yes, the policy covers knee surgery but only after a 2-year waiting period unless it's due to an accident."

### ✅ 2. Application Advisor
- Input: An application or medical summary
- Output:
  - **Approval Likelihood Score (0–100%)**
  - **Actionable Recommendations**
- Analyzes reasoning based on document policies

**Example Output:**
> Likelihood of acceptance: 78%
> 
> Suggestions:
> - Extend policy duration to 6 months
> - Include follow-up care documentation

### 🔍 3. Smart Query-Based Search
- Parse queries like:
  - "46M, knee surgery, Pune, 3-month policy"
- Breaks down into age, condition, location, and duration
- Retrieves and answers contextually from the vector store
