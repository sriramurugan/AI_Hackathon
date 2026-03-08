import streamlit as st
from groq import Groq
import numpy as np
import pandas as pd
import uuid
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import pytesseract
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import sympy as sp
import re

# 1. PAGE CONFIG & NEON THEME
st.set_page_config(page_title="Omni-NCERT Neural Engine", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #05070a; color: #e0e6ed; }
    h1 { background: linear-gradient(90deg, #00ffcc, #0088ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    .formula-box { 
        background: linear-gradient(90deg, #1f2937, #111827); 
        border-left: 5px solid #00ffcc; 
        padding: 15px; border-radius: 10px; margin-bottom: 20px;
        color: #00ffcc; font-family: 'Courier New', Courier, monospace;
        box-shadow: 0 4px 15px rgba(0, 255, 204, 0.2);
    }
    .stChatMessage { border: 1px solid #1f2937; border-radius: 15px; background: #0d1117; }
    </style>
    """, unsafe_allow_html=True)

# 2. ENGINES INITIALIZATION
@st.cache_resource
def load_assets():
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return client, embedder

client, embedder = load_assets()

# 3. GLOBAL STATE
if "chats" not in st.session_state:
    st.session_state.chats = {str(uuid.uuid4()): {"title": "Main Brain", "messages": [], "db": None, "chunks": []}}
    st.session_state.active_id = list(st.session_state.chats.keys())[0]

# --- SIDEBAR: KNOWLEDGE INDEXING ---
with st.sidebar:
    st.title("🧬 Neural Archive")
    if st.button("➕ New Session"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Session", "messages": [], "db": None, "chunks": []}
        st.session_state.active_id = new_id
        st.rerun()

    uploaded_files = st.file_uploader("📂 Upload NCERT Data", type=["pdf", "png", "jpg"], accept_multiple_files=True)
    if st.button("🏗️ Index Knowledge") and uploaded_files:
        with st.spinner("Analyzing Deeply..."):
            raw_text = ""
            for f in uploaded_files:
                if f.type == "application/pdf":
                    pdf = PyPDF2.PdfReader(f)
                    raw_text += "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
                else:
                    raw_text += pytesseract.image_to_string(Image.open(f))
            
            chunks = [raw_text[i:i+800] for i in range(0, len(raw_text), 800)]
            if chunks:
                embeddings = embedder.encode(chunks)
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(np.array(embeddings).astype('float32'))
                st.session_state.chats[st.session_state.active_id]["db"] = index
                st.session_state.chats[st.session_state.active_id]["chunks"] = chunks
                st.success("Knowledge Synced!")

# --- MAIN INTERFACE ---
aid = st.session_state.active_id
active_chat = st.session_state.chats[aid]
tab_chat, tab_lab = st.tabs(["💬 AI Tutor Console", "📊 Pro Visual Lab"])

with tab_chat:
    for m in active_chat["messages"]:
        with st.chat_message(m["role"]): st.markdown(m["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Ask a question or request a graph machi..."):
        active_chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # RAG Context Retrieval
        context = ""
        if active_chat["db"]:
            q_emb = embedder.encode([prompt])
            _, I = active_chat["db"].search(np.array(q_emb).astype('float32'), k=5)
            context = "\n".join([active_chat["chunks"][i] for i in I[0]])

        # 🧠 THE "ULTIMATE UNLOCK" MASTER PROMPT
        sys_msg = f"""You are the Omni-NCERT Neural Engine, a Senior Scientist for 6th-12th Std. 
        
        IMPORTANT CAPABILITIES:
        1. PDF/IMAGE ACCESS: You CAN see the content of uploaded files. The context provided below is extracted directly from the user's files:
           CONTEXT: {context}
        
        2. DYNAMIC GRAPHING: You HAVE the power to render interactive Plotly graphs. Never say "I cannot display images" or "I am a text model".
        
        RULES:
        1. Speak in friendly Tanglish (Machi, Vada, Puriyudha).
        2. STEP-BY-STEP: Whatever the question, explain the logic step-by-step first for students.
        3. TRIGGER CODE: If a graph/visualization is needed, you MUST write Python code in a ```python ... ``` block and assign the result to the variable 'fig'.
        4. Use 'px' or 'go' from plotly for all charts.
        """

        try:
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": sys_msg}] + active_chat["messages"]
            )
            ans = res.choices[0].message.content
            active_chat["messages"].append({"role": "assistant", "content": ans})
            
            with st.chat_message("assistant"):
                st.markdown(ans, unsafe_allow_html=True)
                
                # Dynamic Execution Logic
                code_match = re.search(r'```python\n(.*?)\n```', ans, re.DOTALL)
                if code_match:
                    code_str = code_match.group(1)
                    local_vars = {'np': np, 'go': go, 'px': px, 'sp': sp, 'pd': pd, 'fig': None}
                    try:
                        exec(code_str, globals(), local_vars)
                        if local_vars.get('fig') is not None:
                            st.plotly_chart(local_vars['fig'], use_container_width=True)
                    except Exception as exec_err:
                        st.error(f"Execution Error: {exec_err}")
            st.rerun()
        except Exception as e:
            st.error(f"Brain Sync Error: {e}")

with tab_lab:
    st.header("🔬 3D Lab Simulation")
    st.info("Ask the AI in the chat tab to generate a specific 3D structure or graph, and it will appear here dynamically!")