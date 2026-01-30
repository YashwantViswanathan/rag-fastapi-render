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
import unicodedata
import numpy as np

from rouge_score import rouge_scorer
from sacrebleu import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity

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
    version="2.3.0"
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
# ----------------- UTILITIES ----------------------
# --------------------------------------------------
def clean_generated_text(text: str) -> str:
    if not text:
        return text

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

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip()

def embed_text(text: str):
    return openai_client.embeddings.create(
        model=AZURE_OAI_EMBEDDING_DEPLOYMENT,
        input=text
    ).data[0].embedding

def is_broad_question(question: str) -> bool:
    return any(
        t in question.lower()
        for t in ["overview", "summary", "explain", "all", "policies"]
    )

# --------------------------------------------------
# ---------------- COSMOS RETRIEVAL ----------------
# --------------------------------------------------
def retrieve_chunks(question: str, top_k: int):
    query_embedding = embed_text(question)

    query = """
    SELECT TOP @k c.content
    FROM c
    ORDER BY VectorDistance(c.embedding, @embedding)
    """

    results = container.query_items(
        query=query,
        parameters=[
            {"name": "@k", "value": top_k},
            {"name": "@embedding", "value": query_embedding}
        ],
        enable_cross_partition_query=True
    )

    return [r["content"] for r in results]

# --------------------------------------------------
# -------- CONFIDENCE / GROUNDEDNESS SCORE ----------
# --------------------------------------------------
def compute_confidence_score(answer: str, reference: str, retrieved_chunks: List[str]) -> tuple[float, str]:
    # Embedding similarity
    ans_emb = np.array(embed_text(answer)).reshape(1, -1)
    ref_emb = np.array(embed_text(reference)).reshape(1, -1)
    semantic_sim = cosine_similarity(ans_emb, ref_emb)[0][0]

    # ROUGE-L
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = rouge.score(reference, answer)["rougeL"].fmeasure

    # BLEU
    bleu = sentence_bleu(answer, [reference]).score / 100.0

    # Retrieval strength heuristic
    retrieval_strength = min(len(" ".join(retrieved_chunks)) / 1500, 1.0)

    final_score = (
        0.50 * semantic_sim +
        0.25 * rouge_l +
        0.15 * bleu +
        0.10 * retrieval_strength
    ) * 100

    final_score = round(min(max(final_score, 0), 100), 2)

    if final_score >= 85:
        label = "High"
    elif final_score >= 65:
        label = "Medium"
    else:
        label = "Low"

    return final_score, label

# --------------------------------------------------
# ------------------- RAG PIPE ---------------------
# --------------------------------------------------
def run_rag(question: str):
    broad = is_broad_question(question)
    chunks = retrieve_chunks(question, 30 if broad else 10)

    if not chunks:
        return {
            "answer": "I do not have sufficient information from the provided documents.",
            "confidence_score": 0.0,
            "confidence_label": "Low"
        }

    reference_text = " ".join(chunks[:2])

    messages = [
        {
            "role": "system",
            "content": "Answer strictly using the provided context. Do not add external knowledge."
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

    answer = clean_generated_text(response.choices[0].message.content)

    score, label = compute_confidence_score(answer, reference_text, chunks)

    return {
        "answer": answer,
        "confidence_score": score,
        "confidence_label": label
    }

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
        result = run_rag(q)
        qa_pairs.append({
            "question": q,
            "answer": result["answer"],
            "confidence_score": result["confidence_score"],
            "confidence_label": result["confidence_label"]
        })

    return {"processed": len(qa_pairs)}

@app.get("/download")
def download_csv():
    if not qa_pairs:
        raise HTTPException(status_code=404, detail="No data available")

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["question", "answer", "confidence_score", "confidence_label"]
    )
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
