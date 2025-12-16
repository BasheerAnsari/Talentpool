import json
import re
import docx2txt
from fastapi import UploadFile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM


def clean_text(txt: str):
    return re.sub(r"\s+", " ", txt).strip()


async def load_uploaded_file(file: UploadFile):

    temp_path = f"./temp_{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    low = temp_path.lower()

    if low.endswith(".pdf"):
        docs = PyPDFLoader(temp_path).load()
        text = "\n".join([d.page_content for d in docs])

    elif low.endswith(".docx"):
        text = docx2txt.process(temp_path)

    else:
        docs = TextLoader(temp_path, encoding="utf-8").load()
        text = docs[0].page_content

    return clean_text(text)


def create_vectordb(text):

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = FAISS.from_texts(chunks, embeddings)

    return vectordb


def normalize_skills(value: str):

    if not value or value.lower() == "null":
        return []

    value = value.replace(";", ",").replace("|", ",")
    skills = [s.strip() for s in value.split(",") if s.strip()]

    return skills


PROMPT_FIELDS = """
You are an expert HR Information Extractor.

Extract ONLY the following fields using semantic understanding:
- job_title
- department
- required_skills (comma-separated)
- experience_level
- location
- employment_type

RULES:
1. Use ONLY the meaning from the provided context.
2. If anything is missing → return null.
3. required_skills MUST be comma-separated strings only.
4. STRICT JSON ONLY. NO explanation.

JSON FORMAT:
{{
  "job_title": null,
  "department": null,
  "required_skills": [],
  "experience_level": null,
  "location": null,
  "employment_type": null
}}

CONTEXT:
{context}

RETURN JSON ONLY:
"""


PROMPT_JOBDESC = """
You are an expert HR writer.

Using ONLY the extracted information below:

Job Title: {job_title}
Department: {department}
Skills: {skills}
Experience Level: {experience}
Location: {location}
Employment Type: {employment_type}

Write a clear, polished, professional job description in **8–10 lines**.

RULES:
1. Maintain a professional and HR-appropriate tone.
2. Do NOT add skills or experience not present in the input.
3. Expand naturally using real-world HR writing practices.
4. Return ONLY the job description, no JSON.
"""


def extract_structured_fields(vectordb, model="falcon:instruct"):

    llm = OllamaLLM(model=model, temperature=4)

    docs = vectordb.similarity_search("complete job description meaning", k=2)
    context = " ".join([d.page_content for d in docs])

    prompt = PROMPT_FIELDS.format(context=context)

    result = llm.invoke(prompt)

    return json.loads(result)


def generate_job_description(info, model="falcon:1b"):

    llm = OllamaLLM(model=model, temperature=4)

    prompt = PROMPT_JOBDESC.format(
        job_title=info["job_title"],
        department=info["department"],
        skills=", ".join(info["required_skills"]),
        experience=info["experience_level"],
        location=info["location"],
        employment_type=info["employment_type"],
    )

    return llm.invoke(prompt).strip()


async def extract_jd(file: UploadFile):

    text = await load_uploaded_file(file)
    vectordb = create_vectordb(text)

    # 1. Extract main info
    info = extract_structured_fields(vectordb)

    # Fix skills
    info["required_skills"] = normalize_skills(info["required_skills"])

    # 2. Generate job description
    jd = generate_job_description(info)

    info["job_description"] = jd

    return info