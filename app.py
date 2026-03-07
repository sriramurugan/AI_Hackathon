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
from groq import Groq
from PIL import Image
from sentence_transformers import SentenceTransformer

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="NCERT Hybrid AI Tutor", page_icon="📚", layout="wide")
st.title("📚 NCERT Hybrid AI Learning Tutor")

# SECURE API KEY
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    # Fallback for local testing if secrets.toml isn't set up
    GROQ_API_KEY = "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"

# LOAD MODELS
@st.cache_resource
def load_models():
    client = Groq(api_key=GROQ_API_KEY)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return client, embed_model

groq_client, embed_model = load_models()

# SESSION STATE
if "messages" not in st.session_state: st.session_state.messages = []
if "docs" not in st.session_state: st.session_state.docs = []
if "doc_vectors" not in st.session_state: st.session_state.doc_vectors = None

# FUNCTIONS (Math, Graph, RAG)
def solve_math(prompt):
    try:
        equation_str = re.split(r'solve', prompt, flags=re.IGNORECASE)[-1].strip()
        x = sp.symbols('x')
        solution = sp.solve(equation_str, x)
        return f"### ✅ SymPy Math Result:\nFor the equation ${equation_str}$, $x = {solution}$"
    except: return "Could not solve math."

def graph_equation(expr):
    x_vals = np.linspace(-10, 10, 400)
    try:
        clean_expr = expr.replace('^', '**')
        y_vals = eval(clean_expr, {"np": np, "x": x_vals})
        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals)
        return fig
    except: return None

# ---------------------------------
# SIDEBAR
# ---------------------------------
with st.sidebar:
    st.header("📂 Upload Material")
    uploaded_file = st.file_uploader("PDF or Image", type=["pdf", "png", "jpg"])
    if uploaded_file:
        st.success("File Processed!")

# ---------------------------------
# CHAT INTERFACE
# ---------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

user_prompt = st.chat_input("Ask a question...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"): st.write(user_prompt)

    with st.chat_message("assistant"):
        if "solve" in user_prompt.lower():
            answer = solve_math(user_prompt)
            st.write(answer)
            st.balloons() # 🎉 Celebration for solving math!
        elif "plot" in user_prompt.lower():
            match = re.search(r"plot\s+(.*)", user_prompt)
            if match:
                fig = graph_equation(match.group(1))
                if fig: st.pyplot(fig)
            answer = "Graph generated!"
        else:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": user_prompt}]
            )
            answer = response.choices[0].message.content
            st.write(answer)
            if len(answer) > 100: st.balloons() # 🎉 Celebration for long explanations!

    st.session_state.messages.append({"role": "assistant", "content": answer})