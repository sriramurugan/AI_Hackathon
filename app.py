import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import torch
import pytesseract
import PyPDF2
import faiss
import ollama

from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# PAGE CONFIG
# ------------------------------

st.set_page_config(
    page_title="AI Hybrid Tutor",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Hybrid AI Learning Tutor")

# ------------------------------
# LOAD MODELS
# ------------------------------

@st.cache_resource
def load_models():

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    return embed_model, sentiment


embed_model, sentiment_model = load_models()

# ------------------------------
# SESSION STATE
# ------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "docs" not in st.session_state:
    st.session_state.docs = []

if "doc_vectors" not in st.session_state:
    st.session_state.doc_vectors = None


# ------------------------------
# TOOL ROUTER
# ------------------------------

def detect_task(prompt):

    p = prompt.lower()

    if any(x in p for x in ["graph", "plot", "chart", "visualize"]):
        return "graph"

    if any(x in p for x in ["solve", "equation", "calculate"]):
        return "math"

    if any(x in p for x in ["pdf", "document", "chapter"]):
        return "rag"

    return "chat"


# ------------------------------
# MATH SOLVER
# ------------------------------

def solve_math(prompt):

    try:

        expr = prompt.split("solve")[-1]

        x = sp.symbols("x")

        solution = sp.solve(expr, x)

        return f"Solution: {solution}"

    except:

        return "Could not solve equation."


# ------------------------------
# GRAPH GENERATOR
# ------------------------------

def generate_graph():

    x = np.linspace(0, 20, 100)

    y = 2*x + 5

    fig, ax = plt.subplots()

    ax.plot(x, y)

    ax.set_xlabel("X")

    ax.set_ylabel("Y")

    ax.set_title("Linear Relationship")

    return fig


# ------------------------------
# RAG RETRIEVAL
# ------------------------------

def retrieve_context(query):

    if st.session_state.doc_vectors is None:
        return ""

    q_emb = embed_model.encode([query])

    D, I = st.session_state.doc_vectors.search(
        np.array(q_emb).astype("float32"), 3
    )

    context = ""

    for idx in I[0]:

        context += st.session_state.docs[idx]

    return context


# ------------------------------
# SIDEBAR FILE UPLOAD
# ------------------------------

st.sidebar.header("Upload Learning Material")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF / Image / CSV",
    type=["pdf", "png", "jpg", "jpeg", "csv"]
)

# ------------------------------
# FILE PROCESSING
# ------------------------------

if uploaded_file:

    if uploaded_file.type == "application/pdf":

        reader = PyPDF2.PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            text += page.extract_text()

        chunks = [text[i:i+500] for i in range(0, len(text), 500)]

        embeddings = embed_model.encode(chunks)

        dim = embeddings.shape[1]

        index = faiss.IndexFlatL2(dim)

        index.add(np.array(embeddings).astype("float32"))

        st.session_state.docs = chunks
        st.session_state.doc_vectors = index

        st.sidebar.success("PDF processed")

    elif uploaded_file.type.startswith("image"):

        image = Image.open(uploaded_file)

        text = pytesseract.image_to_string(image)

        chunks = [text]

        embeddings = embed_model.encode(chunks)

        dim = embeddings.shape[1]

        index = faiss.IndexFlatL2(dim)

        index.add(np.array(embeddings).astype("float32"))

        st.session_state.docs = chunks
        st.session_state.doc_vectors = index

        st.sidebar.success("Image processed")

    elif uploaded_file.type == "text/csv":

        df = pd.read_csv(uploaded_file)

        st.write("Dataset preview")

        st.dataframe(df)


# ------------------------------
# CHAT DISPLAY
# ------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])


# ------------------------------
# USER INPUT
# ------------------------------

user_prompt = st.chat_input("Ask anything about studies")

if user_prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt
    })

    task = detect_task(user_prompt)

    if task == "math":

        answer = solve_math(user_prompt)

    elif task == "graph":

        fig = generate_graph()

        st.pyplot(fig)

        answer = "Graph generated."

    else:

        context = retrieve_context(user_prompt)

        prompt = f"""
You are a professional tutor.

Explain clearly.

Context:
{context}

Question:
{user_prompt}

Answer step by step.
"""

        response = ollama.chat(
            model="tinyllama",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"]

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()