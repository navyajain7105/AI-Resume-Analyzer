import os
import uuid
import re
import fitz  # PyMuPDF
import docx
import pandas as pd
import json
import streamlit as st
from io import BytesIO

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv


os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

# ------------------- Config -------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 5

# ------------------- Schema Definition -------------------
class ResumeAnalysis(BaseModel):
    domain: str = Field(..., description="Predicted domain of the resume")
    summary: Optional[str] = Field(..., description="3-line summary")
    strengths: List[str] = Field(..., description="Resume strengths as a list")
    weaknesses: List[str] = Field(..., description="Resume weaknesses as a list")
    score: int = Field(..., description="Overall score out of 100")
    job_match: Optional[str] = Field(default="no", description="Whether resume matches job requirements")

parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)

resume_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional resume evaluator."),
    ("human", """Analyze the following resume and return this JSON:

{format_instructions}

Resume:
{resume_text}
""")
])

evaluation_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an extremely strict technical recruiter. You must respond with ONLY valid JSON, no other text.\n\n"
     "CRITICAL SCORING RULES:\n"
     "- ONLY actual work experience, internships, and real projects count\n"
     "- Academic courses, certifications alone = MAX 20 points\n"
     "- No relevant projects/internships = MAX 30 points\n"
     "- Skills section without proof = 0 points\n"
     "- Art/design/non-tech activities = 0 points for tech roles\n\n"
     "SCORING SCALE:\n"
     "- 0-30: No relevant experience\n"
     "- 31-50: Some relevant coursework but no practical experience\n"
     "- 51-70: Has some projects or internships\n"
     "- 71-85: Good relevant experience\n"
     "- 86-100: Excellent match with proven track record\n\n"
     "BE RUTHLESS. If no real projects exist, score must be under 30."
    ),
     ("human", 
        """Analyze this resume against the job description. Focus ONLY on actual hands-on experience.

Job Description:
{jd_text}

Resume:
{resume_text}

Look for:
1. Actual software projects (GitHub, deployed apps, etc.)
2. Relevant internships or jobs
3. Real technical implementations
4. Ignore: courses, skills lists without context, art projects

Return ONLY this JSON format:
{{
  "domain": "predicted domain",
  "summary": "brief summary focusing on actual experience level",
  "strengths": ["only real strengths with evidence"],
  "weaknesses": ["major gaps in experience"],
  "score": 25
}}

JSON response:""")
])

load_dotenv()

llm = ChatGroq(model_name="llama-3.1-8b-instant")

# ------------------------- Helper Functions -------------------------

def extract_text(file):
    if file.name.endswith(".pdf"):
        try:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            return "\n".join(page.get_text() for page in doc)
        except:
            return ""
    elif file.name.endswith(".docx"):
        try:
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        except:
            return ""
    elif file.name.endswith(".txt"):
        try:
            return file.read().decode("utf-8")
        except:
            return ""
    return ""

def extract_metadata(text: str):
    name_match = re.search(r"Name\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    phone_match = re.search(r"\b\d{10}\b", text)
    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    linkedin_match = re.search(r"(https?://)?(www\.)?linkedin\.com/in/[a-zA-Z0-9\-_/]+", text)
    github_match = re.search(r"(https?://)?(www\.)?github\.com/[a-zA-Z0-9\-_/]+", text)

    return {
        "name": name_match.group(1).strip() if name_match else None,
        "phone": phone_match.group() if phone_match else None,
        "email": email_match.group() if email_match else None,
        "linkedin": linkedin_match.group() if linkedin_match else None,
        "github": github_match.group() if github_match else None,
    }

def get_embedder():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    embedder = get_embedder()
    return FAISS.from_documents(chunks, embedder)

def match_resumes(vectorstore, job_desc, domain_filter, k=TOP_K):
    results = vectorstore.similarity_search(job_desc, k=15)
    filtered = [doc for doc in results if doc.metadata.get("domain") == domain_filter.lower()]
    return filtered[:k]

def analyze_resume(text, domain):
    input_prompt = resume_prompt.format_prompt(
        resume_text=text,
        format_instructions=parser.get_format_instructions()
    )
    output = llm.invoke(input_prompt.to_messages())
    analysis = parser.parse(output.content)
    meta = extract_metadata(text)
    resume_id = f"{domain}_{meta['phone'] or uuid.uuid4().hex[:10]}"
    return analysis, meta, resume_id

def clean_json_response(raw_response: str) -> str:
    """Clean and extract JSON from LLM response"""
    # Remove common prefixes and markdown formatting
    raw_response = raw_response.strip()
    
    # Remove markdown code blocks if present
    if raw_response.startswith('```json'):
        raw_response = raw_response[7:]
    if raw_response.startswith('```'):
        raw_response = raw_response[3:]
    if raw_response.endswith('```'):
        raw_response = raw_response[:-3]
    
    raw_response = raw_response.strip()
    
    # Find JSON object boundaries
    start = raw_response.find('{')
    if start == -1:
        raise ValueError(f"No opening brace found in response: '{raw_response[:100]}...'")
    
    # Find the matching closing brace by counting braces
    brace_count = 0
    end = -1
    for i in range(start, len(raw_response)):
        if raw_response[i] == '{':
            brace_count += 1
        elif raw_response[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i
                break
    
    if end == -1:
        raise ValueError(f"No matching closing brace found. Response starts with: '{raw_response[:200]}...'")
    
    json_str = raw_response[start:end+1]
    return json_str

def evaluate_resume_vs_jd(resume_text: str, jd_text: str):
    """Evaluate resume against job description with improved error handling"""
    
    # Try multiple approaches to get a valid response
    for attempt in range(3):
        try:
            input_prompt = evaluation_prompt.format_prompt(
                resume_text=resume_text,
                jd_text=jd_text
            )
            
            response = llm.invoke(input_prompt.to_messages())
            raw = response.content.strip()
            
            # If response is too short or obviously malformed, try again
            if len(raw) < 20 or not ('{' in raw and '}' in raw):
                if attempt < 2:
                    continue
                else:
                    raise ValueError(f"All attempts failed. Last response: '{raw}'")
            
            # Clean the response
            try:
                cleaned_json = clean_json_response(raw)
            except Exception as clean_error:
                if attempt < 2:
                    continue
                else:
                    raise ValueError(f"JSON cleaning failed after all attempts: {clean_error}")
            
            # Try parsing with manual JSON parsing
            try:
                data = json.loads(cleaned_json)
                
                # Validate required keys
                required_keys = ["domain", "summary", "strengths", "weaknesses", "score"]
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    if attempt < 2:
                        continue
                    else:
                        raise ValueError(f'Missing keys: {missing_keys} in parsed JSON')
                
                # Ensure correct data types
                if not isinstance(data["strengths"], list):
                    if isinstance(data["strengths"], str):
                        data["strengths"] = [s.strip() for s in data["strengths"].split('\n') if s.strip()]
                    else:
                        data["strengths"] = ["Unable to parse strengths"]
                
                if not isinstance(data["weaknesses"], list):
                    if isinstance(data["weaknesses"], str):
                        data["weaknesses"] = [w.strip() for w in data["weaknesses"].split('\n') if w.strip()]
                    else:
                        data["weaknesses"] = ["Unable to parse weaknesses"]
                
                # Ensure score is integer and enforce strict limits
                try:
                    score = int(float(data["score"]))
                    
                    # Additional validation - if score seems too high, cap it
                    resume_lower = resume_text.lower()
                    has_projects = any(keyword in resume_lower for keyword in [
                        'github', 'deployed', 'built', 'developed', 'implemented', 
                        'created', 'project', 'internship', 'work experience'
                    ])
                    
                    has_tech_projects = any(keyword in resume_lower for keyword in [
                        'python', 'java', 'javascript', 'react', 'node', 'sql', 
                        'database', 'api', 'web development', 'software', 'programming'
                    ])
                    
                    # If no clear technical projects found, cap score at 30
                    if not has_projects or not has_tech_projects:
                        if score > 30:
                            score = min(score, 30)
                    
                    data["score"] = score
                    
                except (ValueError, TypeError):
                    data["score"] = 20  # Default low score for parsing errors
                
                # Add job match determination
                data["job_match"] = "yes" if data["score"] >= 60 else "no"
                
                result = ResumeAnalysis(**data)
                return result
                
            except json.JSONDecodeError as json_error:
                if attempt < 2:
                    continue
                else:
                    raise ValueError(f"JSON parsing failed after all attempts: {json_error}\n\nFinal cleaned JSON:\n{cleaned_json}")
            
        except Exception as e:
            if attempt < 2:
                continue
            else:
                break
    
    # Return a default evaluation if all attempts fail
    return ResumeAnalysis(
        domain="unknown",
        summary="Analysis failed due to technical error - multiple parsing attempts unsuccessful",
        strengths=["Unable to evaluate due to technical error"],
        weaknesses=["Technical evaluation error occurred", "Unable to assess actual experience"],
        score=10,  # Very low default score
        job_match="no"
    )

# ------------------- Streamlit App -------------------
st.set_page_config(page_title="AI Resume Analyzer")
st.title("Resume Analyzer + Matcher + Excel Export")

uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
job_desc = st.text_area("Paste Job Description (optional)")

if st.button("Run Analysis"):
    if not uploaded_files:
        st.warning("Please upload resumes.")
    else:
        st.info("Analyzing resumes...")
        all_results = []
        resume_docs = []

        for file in uploaded_files:
            text = extract_text(file)
            if not text.strip():
                st.warning(f"Could not extract text from: {file.name}")
                continue

            inferred_domain = "devops" if "devops" in file.name.lower() else (
                              "datascience" if "ds" in file.name.lower() else (
                              "java" if "java" in file.name.lower() else (
                              "fullstack" if "full" in file.name.lower() else "general")))

            try:
                analysis, meta, resume_id = analyze_resume(text, inferred_domain)
                doc = Document(
                    page_content=text,
                    metadata={
                        "resume_id": resume_id,
                        "file_name": file.name,
                        "domain": inferred_domain,
                        **meta
                    }
                )
                resume_docs.append(doc)

                all_results.append({
                    "Resume ID": resume_id,
                    "File Name": file.name,
                    "Name": meta["name"],
                    "Phone": meta["phone"],
                    "Email": meta["email"],
                    "LinkedIn": meta["linkedin"],
                    "GitHub": meta["github"],
                    "Domain": inferred_domain,
                    "Summary": analysis.summary,
                    "Strengths": "\n".join(analysis.strengths),
                    "Weaknesses": "\n".join(analysis.weaknesses),
                    "Score": analysis.score,
                })

            except Exception as e:
                st.error(f"Error analyzing {file.name}: {e}")

        if job_desc and resume_docs:
            domain_guess = "devops" if "devops" in job_desc.lower() else (
                           "datascience" if "data" in job_desc.lower() else (
                           "java" if "java" in job_desc.lower() else (
                           "fullstack" if "full" in job_desc.lower() else "general")))

            vectorstore = build_vectorstore(resume_docs)
            top_docs = match_resumes(vectorstore, job_desc, domain_guess, k=TOP_K)

            evaluated = {}
            for doc in top_docs:
                try:
                    evaluation = evaluate_resume_vs_jd(doc.page_content, job_desc)
                    evaluated[doc.metadata["resume_id"]] = evaluation
                except Exception as e:
                    pass  # Silently continue if evaluation fails

            # Update results with evaluations
            for row in all_results:
                rid = row["Resume ID"]
                evaluation = evaluated.get(rid)

                if evaluation:
                    row["Strengths"] = "\n".join(evaluation.strengths)
                    row["Weaknesses"] = "\n".join(evaluation.weaknesses)
                    row["Score"] = evaluation.score
                    row["Summary"] = evaluation.summary
                    row["Job Match"] = "✅ Yes" if evaluation.score >= 60 else "❌ No"
                else:
                    row["Job Match"] = "❌ No"

        df = pd.DataFrame(all_results)
        st.success("✅ Analysis Complete")
        
        # Display each resume as an expandable section for better readability
        for idx, row in df.iterrows():
            with st.expander(f"📄 {row['File Name']} - Score: {row['Score']} - {row.get('Job Match', '❌ No')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**👤 Personal Information:**")
                    if row['Name']:
                        st.write(f"**Name:** {row['Name']}")
                    if row['Email']:
                        st.write(f"**Email:** {row['Email']}")
                    if row['Phone']:
                        st.write(f"**Phone:** {row['Phone']}")
                    if row['LinkedIn']:
                        st.write(f"**LinkedIn:** {row['LinkedIn']}")
                    if row['GitHub']:
                        st.write(f"**GitHub:** {row['GitHub']}")
                    
                    st.write(f"**Domain:** {row['Domain']}")
                    st.write(f"**Resume ID:** {row['Resume ID']}")
                
                with col2:
                    st.write("**📊 Evaluation:**")
                    st.write(f"**Score:** {row['Score']}/100")
                    if 'Job Match' in row:
                        st.write(f"**Job Match:** {row['Job Match']}")
                
                st.write("**📝 Summary:**")
                st.write(row['Summary'])
                
                col3, col4 = st.columns(2)
                with col3:
                    st.write("**✅ Strengths:**")
                    strengths = row['Strengths'].split('\n') if isinstance(row['Strengths'], str) else row['Strengths']
                    for strength in strengths:
                        if strength.strip():
                            st.write(f"• {strength.strip()}")
                
                with col4:
                    st.write("**⚠️ Weaknesses:**")
                    weaknesses = row['Weaknesses'].split('\n') if isinstance(row['Weaknesses'], str) else row['Weaknesses']
                    for weakness in weaknesses:
                        if weakness.strip():
                            st.write(f"• {weakness.strip()}")
        
        st.write("---")
        
        # Also show a compact summary table
        st.write("### 📋 Summary Table")
        summary_df = df[['File Name', 'Name', 'Domain', 'Score', 'Job Match']].copy()
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "File Name": st.column_config.TextColumn("File Name", width="large"),
                "Name": st.column_config.TextColumn("Name", width="medium"),
                "Domain": st.column_config.TextColumn("Domain", width="small"),
                "Score": st.column_config.NumberColumn("Score", width="small", format="%d"),
                "Job Match": st.column_config.TextColumn("Job Match", width="small")
            }
        )

        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)

        st.download_button(
            label="📥 Download as Excel",
            data=excel_buffer,
            file_name="resume_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
