import streamlit as st
from groq import Groq
import numpy as np
import pandas as pd
import uuid
import torch
import sympy
from PIL import Image
import pytesseract
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import os

# 1. Advanced Configuration & CSS
st.set_page_config(page_title="NCERT Neural Engine", page_icon="🧬", layout="wide")
st.markdown("<style>.stApp { background: #05070a; color: #00ffcc; }</style>", unsafe_allow_html=True)

# 2. Resource Initialization (The "Heavies")
if "GROQ_API_KEY" not in st.secrets:
    st.error("🔑 Critical Error: GROQ_API_KEY not found.")
    st.stop()

@st.cache_resource
def load_neural_assets():
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    # Local Embedding Model for RAG (Zero-cost, High-speed)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return client, embedder

client, embedder = load_neural_assets()

# 3. Persistent Neural Memory (RAG State)
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
    st.session_state.kb_content = [] # Knowledge Base chunks
if "chats" not in st.session_state:
    st.session_state.chats = {str(uuid.uuid4()): {"title": "Main Brain", "messages": [], "pinned": False}}
    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]

# --- SIDEBAR: KNOWLEDGE INDEXING ---
with st.sidebar:
    st.title("🧬 Neural Index")
    uploaded_files = st.file_uploader("Upload NCERT Books (PDF/IMG)", accept_multiple_files=True)
    
    if st.button("🏗️ Build Knowledge Base") and uploaded_files:
        raw_text = ""
        for f in uploaded_files:
            if f.type == "application/pdf":
                pdf = PyPDF2.PdfReader(f)
                raw_text += "\n".join([p.extract_text() for p in pdf.pages])
            else:
                raw_text += pytesseract.image_to_string(Image.open(f))
        
        # Semantic Chunking
        chunks = [raw_text[i:i+500] for i in range(0, len(raw_text), 500)]
        st.session_state.kb_content = chunks
        
        # Embedding & FAISS Indexing
        embeddings = embedder.encode(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        st.session_state.vector_db = index
        st.success(f"Indexed {len(chunks)} Knowledge Chunks!")

# --- MAIN ENGINE ---
curr_id = st.session_state.current_chat_id
history = st.session_state.chats[curr_id]["messages"]

tab_chat, tab_lab = st.tabs(["💬 Neural Chat (RAG)", "🧪 Visual Simulation Lab"])

with tab_chat:
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Query the NCERT Neural Base..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # RAG RETRIEVAL STEP
        context = ""
        if st.session_state.vector_db:
            q_emb = embedder.encode([prompt])
            D, I = st.session_state.vector_db.search(np.array(q_emb).astype('float32'), k=3)
            context = "\n".join([st.session_state.kb_content[i] for i in I[0]])

        # INFERENCE
        system_prompt = f"""You are a Neural NCERT Tutor. Use this retrieved context if relevant:
        <context>{context}</context>
        Rules: 
        1. Always use step-by-step logic for Class 6-12.
        2. Use LaTeX for math ($$).
        3. If the context contains specific textbook data, prioritize it."""
        
        history.append({"role": "user", "content": prompt})
        try:
            res = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[{"role": "system", "content": system_prompt}] + history,
                temperature=0.1
            )
            ans = res.choices[0].message.content
            with st.chat_message("assistant"):
                st.markdown(ans)
            history.append({"role": "assistant", "content": ans})
        except Exception as e:
            st.error(f"Brain Sync Error: {e}")

with tab_lab:
    st.header("📉 Physics & Math Real-Time Simulator")
    sim = st.selectbox("Select Simulation", ["Projectile Motion", "Acid-Base Titration", "Ohm's Law"])
    
    if sim == "Projectile Motion":
        v0 = st.slider("Initial Velocity (m/s)", 1, 100, 50)
        angle = st.slider("Angle (°)", 0, 90, 45)
        t_max = (2 * v0 * np.sin(np.radians(angle))) / 9.8
        t_range = np.linspace(0, t_max, 100)
        x = v0 * t_range * np.cos(np.radians(angle))
        y = v0 * t_range * np.sin(np.radians(angle)) - 0.5 * 9.8 * t_range**2
        st.line_chart(pd.DataFrame({"Range (m)": x, "Height (m)": y}), x="Range (m)", y=