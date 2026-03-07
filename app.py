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

# 1. PAGE CONFIG (Must be the FIRST Streamlit command)
st.set_page_config(page_title="NCERT Neural Engine v2.0", page_icon="🧬", layout="wide")

# 2. ULTRA-PRO UI CUSTOMIZATION (Neon Dark Theme)
st.markdown("""
    <style>
    .stApp { background: #05070a; color: #e6edf3; }
    h1 { background: linear-gradient(90deg, #00ffcc, #0088ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    section[data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 1px solid #30363d; }
    .stTabs [data-baseweb="tab"] { background-color: #161b22; border-radius: 5px; color: #8b949e; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { color: #00ffcc !important; border-bottom: 2px solid #00ffcc !important; }
    </style>
    """, unsafe_allow_html=True)

# 3. RESOURCE INITIALIZATION
if "GROQ_API_KEY" not in st.secrets:
    st.error("🔑 API Key Missing in Streamlit Secrets!")
    st.stop()

@st.cache_resource
def load_assets():
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return client, embedder

client, embedder = load_assets()

# 4. SESSION STATE
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
    st.session_state.kb_content = [] 
if "chats" not in st.session_state:
    st.session_state.chats = {str(uuid.uuid4()): {"title": "Main Brain", "messages": [], "pinned": False}}
    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]

# --- SIDEBAR: THE UPLOADER ---
with st.sidebar:
    st.title("🧬 Neural Knowledge Index")
    st.success("✅ System v2.0 Online")
    
    uploaded_files = st.file_uploader("📂 Upload NCERT PDFs/Images", 
                                      type=["pdf", "png", "jpg", "jpeg"], 
                                      accept_multiple_files=True)
    
    if st.button("🏗️ Index Knowledge") and uploaded_files:
        with st.spinner("Analyzing Documents..."):
            raw_text = ""
            for f in uploaded_files:
                if f.type == "application/pdf":
                    pdf = PyPDF2.PdfReader(f)
                    for page in pdf.pages:
                        raw_text += page.extract_text() + "\n"
                else:
                    raw_text += pytesseract.image_to_string(Image.open(f))
            
            chunks = [raw_text[i:i+600] for i in range(0, len(raw_text), 600)]
            st.session_state.kb_content = chunks
            embeddings = embedder.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            st.session_state.vector_db = index
            st.sidebar.balloons()
            st.success(f"Indexed {len(chunks)} Chunks!")

# --- MAIN INTERFACE ---
st.title("NCERT Neural Engine")

curr_id = st.session_state.current_chat_id
history = st.session_state.chats[curr_id]["messages"]

tab_chat, tab_lab = st.tabs(["💬 AI Tutor Console", "📊 Science Simulation Lab"])

with tab_chat:
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # RAG Retrieval Logic
        context = ""
        if st.session_state.vector_db:
            q_emb = embedder.encode([prompt])
            D, I = st.session_state.vector_db.search(np.array(q_emb).astype('float32'), k=3)
            context = "\n".join([st.session_state.kb_content[i] for i in I[0]])

        system_prompt = f"""You are a Neural NCERT Tutor. Context: {context}
        Rules: 1. Accurate Class 6-12 answers. 2. Use LaTeX $$ for math. 3. Be concise."""
        
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
            st.error(f"Inference Error: {e}")

with tab_lab:
    st.header("📉 Physics Simulator")
    v0 = st.slider("Velocity", 10, 100, 50)
    ang = st.slider("Angle", 10, 80, 45)
    t_max = (2 * v0 * np.sin(np.radians(ang))) / 9.8
    t = np.linspace(0, t_max, 100)
    x = v0 * t * np.cos(np.radians(ang))
    y = v0 * t * np.sin(np.radians(ang)) - 0.5 * 9.8 * t**2
    st.line_chart(pd.DataFrame({"X": x, "Y": y}), x="X", y="Y")