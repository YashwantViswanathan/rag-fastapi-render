import os
import re
import csv
import io
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from openai import AzureOpenAI
from azure.cosmos import CosmosClient

import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

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

KNOWLEDGE_FILE = os.getenv("KNOWLEDGE_FILE")  # e.g. Godrej_Security_Policy.docx

# --------------------------------------------------
# Initialize FastAPI
# --------------------------------------------------
app = FastAPI(
    title="Batch RAG InfoSec Assistant",
    version="2.1.0"
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
# ----------- KNOWLEDGE INGESTION ------------------
# --------------------------------------------------
def load_knowledge_document(path: str) -> str:
    path = path.lower()

    if path.endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

    elif path.endswith(".docx"):
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        return "\n".join(df.astype(str).fillna("").values.flatten())

    elif path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
        return "\n".join(df.astype(str).fillna("").values.flatten())

    else:
        raise ValueError("Unsupported knowledge document format")

def chunk_text(text: str, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def embed_text(text: str):
    return openai_client.embeddings.create(
        model=AZURE_OAI_EMBEDDING_DEPLOYMENT,
        input=text
    ).data[0].embedding

def ingest_knowledge_document():
    if not KNOWLEDGE_FILE:
        print("No KNOWLEDGE_FILE set. Skipping ingestion.")
        return

    print(f"Ingesting knowledge document: {KNOWLEDGE_FILE}")
    text = load_knowledge_document(KNOWLEDGE_FILE)
    chunks = chunk_text(text)

    for chunk in chunks:
        embedding = embed_text(chunk)
        container.upsert_item({
            "id": str(abs(hash(chunk))),
            "content": chunk,
            "embedding": embedding
        })

    print(f"Ingested {len(chunks)} chunks into Cosmos DB.")

# Run ingestion ONCE at startup
ingest_knowledge_document()

# --------------------------------------------------
# ----------- RAG QUERY PIPELINE -------------------
# --------------------------------------------------
def is_broad_question(question: str) -> bool:
    return any(
        t in question.lower()
        for t in ["overview", "summary", "explain", "all", "policies"]
    )

import unicodedata

def clean_generated_text(text: str) -> str:
    if not text:
        return text

    # Common encoding artifact replacements
    replacements = {
        "â€™": "'",
        "â€œ": '"',
        "â€�": '"',
        "â€“": "-",
        "â€”": "-",
        "â€˜": "'",
        "â€¢": "•",
        "â€¦": "...",
        "Â": "",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Normalize unicode (fix hidden characters)
    text = unicodedata.normalize("NFKC", text)

    # Remove excessive line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing spaces per line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip()


def embed_query(text: str):
    return embed_text(text)

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
        {
            "role": "system",
            "content": "Answer strictly using the provided context."
        },
        {
            "role": "user",
            "content": f"Context:\n{chr(10).join(chunks)}\n\nQuestion:\n{question}"
        }
    ]

    response = openai_client.chat.completions.create(
        model=AZURE_OAI_DEPLOYMENT,
        messages=messages,
        temperature=0.05,
        max_tokens=700
    )

    raw_answer = response.choices[0].message.content
    return clean_generated_text(raw_answer)




# --------------------------------------------------
# ----------- QUESTION FILE PARSING ----------------
# --------------------------------------------------
def extract_from_dataframe(df: pd.DataFrame) -> list[str]:
    questions = []
    for col in df.columns:
        questions.extend(df[col].dropna().astype(str).tolist())
    return [q.strip() for q in questions if q.strip()]

def extract_questions(file: UploadFile) -> list[str]:
    name = file.filename.lower()

    if name.endswith(".csv"):
        return extract_from_dataframe(pd.read_csv(file.file))

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return extract_from_dataframe(pd.read_excel(file.file))

    if name.endswith(".txt"):
        return [l.strip() for l in file.file.read().decode("utf-8").splitlines() if l.strip()]

    if name.endswith(".pdf"):
        reader = PdfReader(file.file)
        text = "".join(p.extract_text() or "" for p in reader.pages)
        return [l.strip() for l in text.splitlines() if l.strip()]

    if name.endswith(".docx"):
        doc = Document(file.file)
        return [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    raise ValueError("Unsupported file type")

# --------------------------------------------------
# ---------------- API ENDPOINTS -------------------
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
        answer = run_rag(q)
        qa_pairs.append({"question": q, "answer": answer})

    return {"processed": len(qa_pairs)}

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
