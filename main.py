import os
import csv
import tempfile
from typing import List

import gradio as gr
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.cosmos import CosmosClient

from PyPDF2 import PdfReader
from docx import Document

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
# Utility functions
# --------------------------------------------------
def embed_text(text: str):
    return openai_client.embeddings.create(
        model=AZURE_OAI_EMBEDDING_DEPLOYMENT,
        input=text
    ).data[0].embedding

def retrieve_chunks(question: str, top_k: int = 5) -> List[str]:
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

def compute_confidence_score(answer: str, true_answer: str):
    ans_emb = np.array(embed_text(answer)).reshape(1, -1)
    ref_emb = np.array(embed_text(true_answer)).reshape(1, -1)
    semantic_sim = cosine_similarity(ans_emb, ref_emb)[0][0]

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = rouge.score(true_answer, answer)["rougeL"].fmeasure

    bleu = sentence_bleu(answer, [true_answer]).score / 100.0

    score = (
        0.65 * semantic_sim +
        0.25 * rouge_l +
        0.10 * bleu
    ) * 100

    score = round(score, 2)

    if score >= 85:
        label = "High"
    elif score >= 65:
        label = "Medium"
    else:
        label = "Low"

    return score, label

def run_rag(question: str):
    chunks = retrieve_chunks(question)

    if not chunks:
        return "No relevant knowledge found.", 0.0, "Low"

    true_answer = chunks[0]

    messages = [
        {
            "role": "system",
            "content": "Answer strictly using the provided context."
        },
        {
            "role": "user",
            "content": f"Context:\n{true_answer}\n\nQuestion:\n{question}"
        }
    ]

    response = openai_client.chat.completions.create(
        model=AZURE_OAI_DEPLOYMENT,
        messages=messages,
        temperature=0.05,
        max_tokens=600
    )

    answer = response.choices[0].message.content.strip()
    score, label = compute_confidence_score(answer, true_answer)

    return answer, score, label

# --------------------------------------------------
# File parsing
# --------------------------------------------------
def extract_questions(file_path: str) -> List[str]:
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        return df.iloc[:, 0].dropna().astype(str).tolist()

    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
        return df.iloc[:, 0].dropna().astype(str).tolist()

    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]

    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = "".join(p.extract_text() or "" for p in reader.pages)
        return [l.strip() for l in text.splitlines() if l.strip()]

    if file_path.endswith(".docx"):
        doc = Document(file_path)
        return [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    raise ValueError("Unsupported file type")

# --------------------------------------------------
# Gradio logic
# --------------------------------------------------
def process_file(file):
    questions = extract_questions(file.name)

    rows = []
    for q in questions:
        answer, score, label = run_rag(q)
        rows.append([q, answer, score, label])

    df = pd.DataFrame(
        rows,
        columns=["Question", "Answer", "Confidence Score", "Label"]
    )

    # Save CSV for download
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)

    return df, tmp.name

# --------------------------------------------------
# Gradio UI
# --------------------------------------------------
with gr.Blocks(title="Response Generation AI Agent") as demo:
    gr.Markdown("## Response Generation AI Agent")
    gr.Markdown("### Trial Version 3")

    with gr.Row():
        file_input = gr.File(label="Upload Question File")
        run_btn = gr.Button("Run RAG")

    output_table = gr.Dataframe(
        headers=["Question", "Answer", "Confidence Score", "Label"],
        wrap=True
    )

    csv_output = gr.File(label="Download Output CSV")

    run_btn.click(
        fn=process_file,
        inputs=file_input,
        outputs=[output_table, csv_output]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
