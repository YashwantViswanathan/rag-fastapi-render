import os
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
        color = "yellow"
    elif score >= 65:
        label = "Medium"
        color = "orange"
    else:
        label = "Low"
        color = "red"

    return score, label, color

def run_rag(question: str):
    chunks = retrieve_chunks(question)

    if not chunks:
        return "No relevant knowledge found.", 0.0, "Low", "red"

    true_answer = chunks[0]

    response = openai_client.chat.completions.create(
        model=AZURE_OAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Answer strictly using the provided context."},
            {"role": "user", "content": f"Context:\n{true_answer}\n\nQuestion:\n{question}"}
        ],
        temperature=0.05,
        max_tokens=600
    )

    answer = response.choices[0].message.content.strip()
    return (*compute_confidence_score(answer, true_answer), answer)

# --------------------------------------------------
# File parsing
# --------------------------------------------------
def extract_questions(file_path: str) -> List[str]:
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    elif file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = "".join(p.extract_text() or "" for p in reader.pages)
        return [l.strip() for l in text.splitlines() if l.strip()]
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    else:
        raise ValueError("Unsupported file type")

    questions = []
    for col in df.columns:
        questions.extend(df[col].dropna().astype(str).tolist())

    return [q.strip() for q in questions if q.strip()]

# --------------------------------------------------
# Gradio processing
# --------------------------------------------------
def process_file(file):
    questions = extract_questions(file.name)

    if not questions:
        raise gr.Error("No questions found in the uploaded file.")

    rows = []
    for q in questions:
        score, label, color, answer = run_rag(q)
        rows.append([
            q,
            answer,
            f"<span style='color:{color}; font-weight:bold'>{score}</span>",
            f"<span style='color:{color}; font-weight:bold'>{label}</span>"
        ])

    df = pd.DataFrame(
        rows,
        columns=["Question", "Answer", "Confidence Score", "Label"]
    )

    output_path = os.path.join(tempfile.gettempdir(), "Generated_Responses.csv")
    df.to_csv(output_path, index=False)

    return df, output_path

# --------------------------------------------------
# Custom CSS (Blue–Black theme)
# --------------------------------------------------
custom_css = """
/* Page background */
body {
    background: linear-gradient(135deg, #0b2a4a 0%, #020617 55%, #000000 100%);
}

/* Gradio root container */
.gradio-container {
    background: transparent !important;
}

/* Section wrappers (keep transparent so blue shows) */
.wrap,
.contain,
.app {
    background: transparent !important;
}

/* Header blocks */
h1, h2, h3 {
    color: #e5e7eb;
}

/* Input + output cards ONLY */
.block:has(input),
.block:has(button),
.block:has(table),
.block:has(.dataframe) {
    background: #000000 !important;
    border-radius: 12px;
    padding: 14px;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.6);
}

/* Dataframe styling */
table {
    background: #000000 !important;
}

/* Improve table row contrast */
tr:nth-child(even) {
    background-color: #020617;
}
"""

# --------------------------------------------------
# Gradio UI
# --------------------------------------------------
with gr.Blocks(css=custom_css, title="Response Generation AI Agent") as demo:
    gr.Markdown("## Response Generation AI Agent")
    gr.Markdown("### Trial Version 3")

    with gr.Row():
        file_input = gr.File(label="Upload Question File")
        run_btn = gr.Button("Run RAG")

    output_table = gr.Dataframe(
        headers=["Question", "Answer", "Confidence Score", "Label"],
        datatype=["str", "str", "html", "html"],
        wrap=True
    )

    csv_output = gr.File(label="Download Output CSV")

    run_btn.click(
        fn=process_file,
        inputs=file_input,
        outputs=[output_table, csv_output]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=port)
