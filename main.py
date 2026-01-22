import os
import re
import csv
import io
from typing import List

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import AzureOpenAI
from azure.cosmos import CosmosClient

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
    title="Hybrid RAG InfoSec Assistant",
    description="FastAPI-based Hybrid RAG using Azure OpenAI + Cosmos DB",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# --------------------------------------------------
# Initialize clients (singleton)
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
# In-memory Q&A store (per app instance)
# --------------------------------------------------
qa_pairs: List[dict] = []

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def is_broad_question(question: str) -> bool:
    q = question.lower()
    broad_terms = [
        "tell me about",
        "overview",
        "summary",
        "all",
        "entire",
        "policies",
        "explain"
    ]
    return any(term in q for term in broad_terms)

def extract_keywords(question: str):
    tokens = re.findall(r"\b[a-zA-Z]{4,}\b", question.lower())
    stopwords = {
        "tell", "about", "what", "which", "policy", "policies",
        "explain", "overview", "summary", "describe"
    }
    return [t for t in tokens if t not in stopwords][:3]

def embed_query(text: str):
    response = openai_client.embeddings.create(
        model=AZURE_OAI_EMBEDDING_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding

def retrieve_chunks_hybrid(question: str, top_k: int):
    embedding = embed_query(question)
    keywords = extract_keywords(question)

    params = [
        {"name": "@k", "value": top_k},
        {"name": "@embedding", "value": embedding}
    ]

    where_clause = ""
    if keywords:
        conditions = []
        for i, kw in enumerate(keywords):
            pname = f"@kw{i}"
            conditions.append(f"CONTAINS(c.content, {pname}, true)")
            params.append({"name": pname, "value": kw})
        where_clause = "WHERE " + " OR ".join(conditions)

    query = f"""
    SELECT TOP @k c.content
    FROM c
    {where_clause}
    ORDER BY VectorDistance(c.embedding, @embedding)
    """

    results = list(container.query_items(
        query=query,
        parameters=params,
        enable_cross_partition_query=True
    ))

    # Fallback to pure vector search
    if len(results) < max(3, top_k // 3):
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

    return [item["content"] for item in results]

def build_specific_prompt(context_chunks, question):
    context = "\n\n".join(context_chunks)
    return [
        {
            "role": "system",
            "content": (
                "You are an enterprise information security assistant.\n"
                "Answer ONLY using the provided context.\n"
                "Be precise and policy-accurate.\n"
                "If the answer is not present, say:\n"
                "'I do not have sufficient information from the provided documents.'"
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }
    ]

def build_overview_prompt(context_chunks, question):
    context = "\n\n".join(context_chunks)
    return [
        {
            "role": "system",
            "content": (
                "You are an enterprise information security analyst.\n"
                "Summarize all policy areas found in the provided excerpts.\n"
                "Base your response ONLY on the excerpts."
            )
        },
        {
            "role": "user",
            "content": f"Policy excerpts:\n{context}\n\nRequest:\n{question}"
        }
    ]

# --------------------------------------------------
# Request / Response models
# --------------------------------------------------
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# --------------------------------------------------
# API Endpoints
# --------------------------------------------------
@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    question = payload.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    broad = is_broad_question(question)
    top_k = 30 if broad else 10

    chunks = retrieve_chunks_hybrid(question, top_k)

    if broad and len(chunks) < 12:
        chunks = retrieve_chunks_hybrid(question, 45)

    if not chunks:
        answer = "I do not have sufficient information from the provided documents."
    else:
        messages = (
            build_overview_prompt(chunks, question)
            if broad
            else build_specific_prompt(chunks, question)
        )

        response = openai_client.chat.completions.create(
            model=AZURE_OAI_DEPLOYMENT,
            temperature=0.05,
            max_tokens=900,
            messages=messages
        )

        answer = response.choices[0].message.content

    qa_pairs.append({
        "question": question,
        "answer": answer
    })

    return {"answer": answer}

@app.get("/download")
def download_qa_csv():
    if not qa_pairs:
        raise HTTPException(status_code=404, detail="No Q&A data available")

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["question", "answer"])
    writer.writeheader()
    writer.writerows(qa_pairs)

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=rag_question_answer_log.csv"}
    )

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

