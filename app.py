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
    if not text or not isinstance(text, str) or not text.strip():
        raise ValueError("Invalid text passed for embedding")

    return openai_client.embeddings.create(
        model=AZURE_OAI_EMBEDDING_DEPLOYMENT,
        input=text.strip()
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
        return {
            "answer": "No relevant knowledge found.",
            "score": 0.0,
            "label": "Low",
            "color": "red"
        }

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

    if not answer:
        return {
            "answer": "No answer generated.",
            "score": 0.0,
            "label": "Low",
            "color": "red"
        }

    score, label, color = compute_confidence_score(answer, true_answer)

    return {
        "answer": answer,
        "score": score,
        "label": label,
        "color": color
    }

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

    ui_rows = []
    csv_rows = []

    for q in questions:
        result = run_rag(q)

    # UI row (HTML)
        ui_rows.append([
            q,
            result["answer"],
            f"<span style='color:{result['color']}; font-weight:bold'>{result['score']}</span>",
            f"<span style='color:{result['color']}; font-weight:bold'>{result['label']}</span>"
    ])

    # CSV row (plain)
        csv_rows.append([
            q,
            result["answer"],
            result["score"],
            result["label"]
        ])


    # UI DataFrame
    ui_df = pd.DataFrame(
        ui_rows,
        columns=["Question", "Answer", "Confidence Score", "Label"]
    )

    # CSV DataFrame (NO HTML)
    csv_df = pd.DataFrame(
        csv_rows,
        columns=["Question", "Answer", "Confidence Score", "Label"]
    )

    output_path = os.path.join(tempfile.gettempdir(), "Generated_Responses.csv")
    csv_df.to_csv(output_path, index=False)

    return ui_df, output_path

# --------------------------------------------------
# Custom CSS (Blue–Black theme)
# --------------------------------------------------
custom_css = """
/* =====================
   RESPECT LIGHT / DARK MODE
   ===================== */

:root {
    --card-bg: var(--background-fill-secondary);
    --border-color: var(--border-color-primary);
    --text-color: var(--body-text-color);
    --hover-bg: var(--neutral-200);
}

@media (prefers-color-scheme: dark) {
    :root {
        --hover-bg: var(--neutral-800);
    }
}

/* =====================
   GENERAL LAYOUT CLEANUP
   ===================== */

.gradio-container {
    background: var(--background-fill-primary) !important;
}

/* Cards / blocks */
.block {
    background: var(--card-bg) !important;
    border-radius: 12px;
    padding: 12px;
    border: 1px solid var(--border-color);
}

/* Headings */
h1, h2, h3 {
    color: var(--text-color) !important;
}

/* =====================
   INPUTS & BUTTONS
   ===================== */

input, textarea {
    background: var(--background-fill-primary) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
}

button {
    border-radius: 10px;
    font-weight: 600;
}

/* =====================
   TABLE STYLING
   ===================== */

table {
    background: transparent !important;
    color: var(--text-color) !important;
    border-collapse: collapse;
}

thead th {
    font-weight: 600;
    border-bottom: 2px solid var(--border-color);
    padding: 12px;
}

tbody tr {
    transition: background 0.15s ease, transform 0.15s ease;
}

tbody tr:hover {
    background: var(--hover-bg) !important;
    transform: scale(1.01);
}

td {
    padding: 12px;
    vertical-align: top;
    border-bottom: 1px solid var(--border-color);
}

/* =====================
   CONFIDENCE COLORS
   ===================== */

.conf-high {
    color: #16a34a;
    font-weight: 700;
}

.conf-medium {
    color: #f59e0b;
    font-weight: 700;
}

.conf-low {
    color: #dc2626;
    font-weight: 700;
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
