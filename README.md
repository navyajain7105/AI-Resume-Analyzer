# 🧠 AI Resume Analyzer + Matcher

This Streamlit app uses **LLMs (Mistral via Groq)**, **LangChain**, and **FAISS** to intelligently:

- 📄 Analyze uploaded resumes (PDF/DOCX/TXT)
- 🧠 Extract metadata (name, phone, email, LinkedIn, GitHub)
- 📊 Generate a structured summary, strengths, weaknesses, and a strict evaluation score
- 🔍 Match resumes to a given **job description**
- 📥 Export results as Excel for easy review

---

## 🚀 Features

✅ Upload multiple resumes  
✅ Auto domain detection (DevOps, Data Science, Java, Fullstack)  
✅ Use **HuggingFace embeddings** for semantic similarity  
✅ Evaluate resumes strictly against a JD  
✅ Export structured results with strengths, weaknesses, and scores  
✅ Built with **LangChain v0.2+**, **FAISS**, **PyMuPDF**, and **Streamlit**

---

## 🛠️ Tech Stack

- [Python 3.10+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Groq + Mistral](https://groq.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- PyMuPDF (`fitz`) for PDF parsing
- `python-docx` for DOCX parsing
- `pydantic` for structured output
- `openpyxl` for Excel export

---

## 🧪 How it works

1. Upload resumes (`.pdf`, `.docx`, or `.txt`)
2. (Optional) Paste a job description
3. Click "Run Analysis"
4. The app will:
    - Extract metadata (name, email, LinkedIn, GitHub)
    - Predict the resume domain
    - Analyze each resume using a structured LLM prompt
    - Match top resumes using FAISS vector similarity (if JD is given)
    - Strictly evaluate the top-k matches against JD
5. View and export all results as an Excel file

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/navyajain7105/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
