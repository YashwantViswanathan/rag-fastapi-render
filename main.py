import os
import re
import csv
import io
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv


from openai import AzureOpenAI
from azure.cosmos import CosmosClient

import pandas as pd
import PyPDF2
from docx import Document
from pypdf import PDFReader

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

AZURE_OAI_ENDPOINT = os.getenv("AZURE_OAI_ENDPOINT")
AZURE_OAI_KEY = os.getenv("AZURE_OAI_KEY")
AZURE_OAI_DEPLOYMENT = os.getenv("AZURE_OAI_DEPLOYMENT")
AZURE_OAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OAI_EMBEDDING_DEPLOYMENT")

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER")

# --------------------------------------------------
# Initialize FastAPI
# --------------------------------------------------
app = FastAPI(
    title="Batch RAG InfoSec Assistant",
    version="2.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------------------------------------
# Clients
# --------------------------------------------------
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OAI_ENDPOINT,
    api_key=AZURE_OAI_KEY,
    api_version="2024-02-15-preview"
)

cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
db = cosmos_client.get_database_client(COSMOS_DATABASE)
container = db.get_container_client(COSMOS_CONTAINER)

# --------------------------------------------------
# In-memory Q&A store
# --------------------------------------------------
qa_pairs: List[dict] = []

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def is_broad_question(question: str) -> bool:
    return any(
        t in question.lower()
        for t in ["overview", "summary", "explain", "all", "policies"]
    )

def extract_keywords(question: str):
    tokens = re.findall(r"\b[a-zA-Z]{4,}\b", question.lower())
    return tokens[:3]

def embed_query(text: str):
    return openai_client.embeddings.create(
        model=AZURE_OAI_EMBEDDING_DEPLOYMENT,
        input=text
    ).data[0].embedding

def retrieve_chunks(question: str, top_k: int):
    embedding = embed_query(question)
    query = """
    SELECT TOP @k c.content
    FROM c
    ORDER BY VectorDistance(c.embedding, @embedding)
    """
    results = container.query_items(
        query=query,
        parameters=[
            {"name": "@k", "value": top_k},
            {"name": "@embedding", "value": embedding}
        ],
        enable_cross_partition_query=True
    )
    return [r["content"] for r in results]

def run_rag(question: str) -> str:
    broad = is_broad_question(question)
    chunks = retrieve_chunks(question, 30 if broad else 10)

    if not chunks:
        return "I do not have sufficient information from the provided documents."

    messages = [
        {"role": "system", "content": "Answer strictly from the given context."},
        {"role": "user", "content": f"Context:\n{chr(10).join(chunks)}\n\nQ: {question}"}
    ]

    response = openai_client.chat.completions.create(
        model=AZURE_OAI_DEPLOYMENT,
        messages=messages,
        temperature=0.05,
        max_tokens=700
    )

    return response.choices[0].message.content

def extract_from_dataframe(df: pd.DataFrame) -> list[str]:
    # Flatten all columns into a single list
    questions = []

    for col in df.columns:
        col_data = df[col].dropna().astype(str).tolist()
        questions.extend(col_data)

    # Clean & deduplicate
    questions = [q.strip() for q in questions if q.strip()]
    return questions


# --------------------------------------------------
# File parsing
# --------------------------------------------------
def extract_questions(file: UploadFile) -> list[str]:
    filename = file.filename.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(file.file)
        return extract_from_dataframe(df)

    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(file.file)
        return extract_from_dataframe(df)

    elif filename.endswith(".txt"):
        content = file.file.read().decode("utf-8")
        return [line.strip() for line in content.splitlines() if line.strip()]

    elif filename.endswith(".pdf"):
        reader = PdfReader(file.file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return [line.strip() for line in text.splitlines() if line.strip()]

    elif filename.endswith(".docx"):
        document = Document(file.file)
        questions = [
            para.text.strip()
            for para in document.paragraphs
            if para.text.strip()
        ]
        return questions


    else:
        raise ValueError("Unsupported file type")


# --------------------------------------------------
# API endpoints
# --------------------------------------------------
@app.post("/upload")
def upload_questions(file: UploadFile = File(...)):
    qa_pairs.clear()

    try:
        questions = extract_questions(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    if not questions:
        raise HTTPException(status_code=400, detail="No questions found in file")

    for q in questions:
        if not q.strip():
            continue
        answer = run_rag(q)
        qa_pairs.append({"question": q, "answer": answer})

    return {
        "processed": len(qa_pairs),
        "status": "success"
    }


@app.get("/download")
def download_csv():
    if not qa_pairs:
        raise HTTPException(status_code=404, detail="No data available")

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["question", "answer"])
    writer.writeheader()
    writer.writerows(qa_pairs)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_qna_output.csv"}
    )

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")
