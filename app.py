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
import networkx as nx

# 1. PAGE CONFIG & NEON THEME
st.set_page_config(page_title="Omni-Neural Discovery Engine", page_icon="🕸️", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #05070a; color: #e0e6ed; }
    h1 { background: linear-gradient(90deg, #00ffcc, #0088ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    .stChatMessage { border: 1px solid #1f2937; border-radius: 15px; background: #0d1117; margin-bottom: 10px; }
    .feedback-btn { border-radius: 20px; border: 1px solid #00ffcc; background: transparent; color: #00ffcc; padding: 2px 10px; font-size: 0.8rem; cursor: pointer; }
    </style>
    """, unsafe_allow_html=True)

# 2. ENGINES INITIALIZATION
@st.cache_resource
def load_assets():
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return client, embedder

client, embedder = load_assets()

# IteratorsHQ Principle: Data Cleaning Layer
def clean_text_data(text):
    text = re.sub(r'\s+', ' ', text) # Remove extra whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # Remove non-ascii noise
    return text.strip()

# 3. GLOBAL STATE (Long-Term Memory + Graph Storage)
if "chats" not in st.session_state:
    st.session_state.chats = {
        str(uuid.uuid4()): {
            "messages": [], 
            "db": None, 
            "chunks": [], 
            "graph": nx.Graph(),
            "quality_feedback": []
        }
    }
    st.session_state.active_id = list(st.session_state.chats.keys())[0]

# --- SIDEBAR: KNOWLEDGE PIPELINE (GraphRAG + Quality Check) ---
with st.sidebar:
    st.title("🕸️ Discovery Archive")
    if st.button("➕ New Neural Session"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"messages": [], "db": None, "chunks": [], "graph": nx.Graph(), "quality_feedback": []}
        st.session_state.active_id = new_id
        st.rerun()

    uploaded_files = st.file_uploader("📂 Data Ingestion (PDF/Images)", type=["pdf", "png", "jpg"], accept_multiple_files=True)
    if st.button("🏗️ Build Knowledge Graph") and uploaded_files:
        with st.spinner("IteratorsHQ-Style Cleaning & GraphRAG Indexing..."):
            raw_text = ""
            for f in uploaded_files:
                if f.type == "application/pdf":
                    pdf = PyPDF2.PdfReader(f)
                    raw_text += "\n".join([clean_text_data(p.extract_text()) for p in pdf.pages if p.extract_text()])
                else:
                    raw_text += clean_text_data(pytesseract.image_to_string(Image.open(f)))
            
            # Semantic Indexing (Vector)
            chunks = [raw_text[i:i+600] for i in range(0, len(raw_text), 600)]
            embeddings = embedder.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            
            # Relationship Indexing (Microsoft GraphRAG Style)
            G = nx.Graph()
            for chunk in chunks[:15]: # Process chunks for relationship discovery
                entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', chunk)
                for i in range(len(entities)-1):
                    G.add_edge(entities[i], entities[i+1])
            
            st.session_state.chats[st.session_state.active_id]["db"] = index
            st.session_state.chats[st.session_state.active_id]["chunks"] = chunks
            st.session_state.chats[st.session_state.active_id]["graph"] = G
            st.success("Discovery Engine Ready!")

# --- MAIN INTERFACE ---
aid = st.session_state.active_id
active_chat = st.session_state.chats[aid]
tab_chat, tab_map = st.tabs(["💬 Augmented Intelligence", "🕸️ Knowledge Graph"])

with tab_chat:
    for i, m in enumerate(active_chat["messages"]):
        with st.chat_message(m["role"]):
            st.markdown(m["content"], unsafe_allow_html=True)
            if m["role"] == "assistant":
                if st.button("👍 Correct", key=f"up_{i}"): st.toast("Quality Label: Positive")
                if st.button("👎 Wrong", key=f"dn_{i}"): st.toast("Quality Label: Negative")

    if prompt := st.chat_input("Analyze, Solve, or Visualize..."):
        active_chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # 1. RETRIEVAL (Vector Similarity + Relation Context)
        context = ""
        if active_chat["db"]:
            _, I = active_chat["db"].search(np.array(embedder.encode([prompt])).astype('float32'), k=4)
            context = "\n".join([active_chat["chunks"][i] for i in I[0]])

        # 2. THE MASTER SYSTEM PROMPT (Combines Microsoft GraphRAG + IteratorsHQ Quality + NCERT Tutor)
        sys_msg = f"""You are the Omni-Neural Engine. You use Microsoft GraphRAG logic for discovery and follow IteratorsHQ principles for data quality.
        
        KNOWLEDGE CONTEXT: {context}
        
        INSTRUCTIONS:
        1. Speak in friendly Tanglish (Machi, Vada).
        2. PEDAGOGY: Solve any 6th-12th NCERT problem step-by-step.
        3. DISCOVERY: Explain not just the answer, but the RELATIONSHIPS between concepts in the context.
        4. DYNAMIC VISUALS: If a graph/3D plot is needed, write Plotly code in a ```python ... ``` block. Use 'fig' as the variable name.
        5. If context is 'Garbage' or missing, ask the user to upload clean data.
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
                code_match = re.search(r'```python\n(.*?)\n```', ans, re.DOTALL)
                if code_match:
                    local_vars = {'np': np, 'go': go, 'px': px, 'sp': sp, 'pd': pd, 'fig': None}
                    exec(code_match.group(1), globals(), local_vars)
                    if local_vars['fig']: st.plotly_chart(local_vars['fig'])
            st.rerun()
        except Exception as e: st.error(f"Engine Fault: {e}")

with tab_map:
    st.header("🕸️ Neural Concept Map (GraphRAG View)")
    G = active_chat["graph"]
    if len(G.nodes) > 0:
        pos = nx.spring_layout(G)
        node_trace = go.Scatter(x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
                                mode='markers+text', text=list(G.nodes()), textposition="bottom center",
                                marker=dict(size=12, color='#00ffcc'))
        fig = go.Figure(data=[node_trace], layout=go.Layout(template="plotly_dark", showlegend=False))
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("Machi, index knowledge to see the concept graph!")