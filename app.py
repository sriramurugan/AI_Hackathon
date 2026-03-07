import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import torch
import pytesseract
import PyPDF2
import faiss
import re
from groq import Groq  # Swapped Ollama for Groq for speed
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------------------------------
# PAGE CONFIG & STYLING
# ---------------------------------
st.set_page_config(page_title="NCERT Hybrid AI Tutor", page_icon="📚", layout="wide")
st.title("📚 NCERT Hybrid AI Learning Tutor")
st.markdown("*Your intelligent companion for NCERT Science & Math*")

# ---------------------------------
# LOAD MODELS (Caching for i3 performance)
# ---------------------------------
@st.cache_resource
def load_models():
    # Groq Client setup
    client = Groq(api_key="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")
    # Embedding model for RAG
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return client, embed_model

groq_client, embed_model = load_models()

# ---------------------------------
# SESSION STATE
# ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs" not in st.session_state:
    st.session_state.docs = []
if "doc_vectors" not in st.session_state:
    st.session_state.doc_vectors = None

# ---------------------------------
# LOGIC FUNCTIONS
# ---------------------------------

def detect_task(prompt):
    p = prompt.lower()
    if any(x in p for x in ["plot", "graph", "draw"]): return "graph"
    if any(x in p for x in ["solve", "equation", "integrate", "differentiate"]): return "math"
    return "chat"

def solve_math(prompt):
    try:
        # Extracts everything after 'solve'
        equation_str = re.split(r'solve', prompt, flags=re.IGNORECASE)[-1].strip()
        x = sp.symbols('x')
        # Simple parsing for NCERT level algebra
        solution = sp.solve(equation_str, x)
        return f"### SymPy Math Result:\nFor the equation ${equation_str}$, the value of $x$ is: **{solution}**"
    except Exception as e:
        return f"I couldn't parse that math problem. Please try: 'solve x**2 - 4'"

def graph_equation(expr):
    x_vals = np.linspace(-10, 10, 400)
    try:
        # Security: replacing '^' with '**' for python eval
        clean_expr = expr.replace('^', '**')
        y_vals = eval(clean_expr, {"np": np, "x": x_vals, "sin": np.sin, "cos": np.cos, "tan": np.tan, "sqrt": np.sqrt})
        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, color='royalblue', linewidth=2)
        ax.axhline(0, color='black', lw=1)
        ax.axvline(0, color='black', lw=1)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(f"Graph of y = {expr}")
        return fig
    except:
        return None

def retrieve_context(query):
    if st.session_state.doc_vectors is None or not st.session_state.docs:
        return ""
    q_emb = embed_model.encode([query])
    D, I = st.session_state.doc_vectors.search(np.array(q_emb).astype("float32"), 3)
    context = "\n".join([st.session_state.docs[idx] for idx in I[0] if idx != -1])
    return context

# ---------------------------------
# SIDEBAR: FILE PROCESSING
# ---------------------------------
with st.sidebar:
    st.header("📂 Study Materials")
    uploaded_file = st.file_uploader("Upload NCERT PDF/Image", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file:
        with st.spinner("Indexing your material..."):
            text_data = ""
            if uploaded_file.type == "application/pdf":
                reader = PyPDF2.PdfReader(uploaded_file)
                text_data = "".join([page.extract_text() for page in reader.pages])
            else:
                image = Image.open(uploaded_file)
                text_data = pytesseract.image_to_string(image)

            # Chunking and Vectorizing
            chunks = [text_data[i:i+500] for i in range(0, len(text_data), 500)]
            embeddings = embed_model.encode(chunks)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(np.array(embeddings).astype("float32"))
            
            st.session_state.docs = chunks
            st.session_state.doc_vectors = index
            st.success("Context loaded! Ask me anything about this file.")

# ---------------------------------
# MAIN CHAT INTERFACE
# ---------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_prompt = st.chat_input("Ask a question, e.g., 'What is Photosynthesis?' or 'plot x**2'")

if user_prompt:
    st.chat_message("user").write(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    task = detect_task(user_prompt)
    
    with st.chat_message("assistant"):
        if task == "math":
            answer = solve_math(user_prompt)
            st.write(answer)
        
        elif task == "graph":
            match = re.search(r"(?:plot|graph)\s+(.*)", user_prompt, re.IGNORECASE)
            if match:
                expr = match.group(1)
                fig = graph_equation(expr)
                if fig:
                    st.pyplot(fig)
                    answer = f"Generated graph for ${expr}$"
                else: answer = "Could not plot that. Check the formula!"
            else: answer = "Try: 'plot x**2'"
            st.write(answer)
            
        else:
            # LLM + RAG logic using Groq
            context = retrieve_context(user_prompt)
            full_prompt = f"Context: {context}\n\nQuestion: {user_prompt}\n\nHelp the student understand step-by-step."
            
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": full_prompt}]
                )
                answer = response.choices[0].message.content
                st.write(answer)
            except Exception as e:
                st.error("Groq connection error. Check your API key!")
                answer = "Error connecting to AI."

    st.session_state.messages.append({"role": "assistant", "content": answer})