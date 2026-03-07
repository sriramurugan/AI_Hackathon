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
                    raw_text += "\n".join([p.extract_text() for p in pdf.pages])
                else:
                    raw_text += pytesseract.image_to_string(Image.open(f))
            chunks = [raw_text[i:i+800] for i in range(0, len(raw_text), 800)]
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

    if prompt := st.chat_input("Complex calculation or Chemistry theory kelu machi..."):
        active_chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # RAG Context Retrieval
        context = ""
        if active_chat["db"]:
            q_emb = embedder.encode([prompt])
            _, I = active_chat["db"].search(np.array(q_emb).astype('float32'), k=4)
            context = "\n".join([active_chat["chunks"][i] for i in I[0]])

        # 🧠 THE "PAKKAVA" MASTER PROMPT
        sys_msg = f"""You are a Senior NCERT Scientist. Context: {context}.
        RULES:
        1. Speak in friendly Tanglish.
        2. START with <div class='formula-box'>📌 **FORMULA KEY:** [LaTeX]</div> for Physics/Maths.
        3. For Chemistry: Detail the Hybridization (sp, sp2, sp3, dsp2), 3D Geometry, and Lone pairs.
        4. Calculation: Solve step-by-step.
        5. Visuals: If needed, trigger plot logic by mentioning 'Plotting graph'."""

        try:
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": sys_msg}] + active_chat["messages"]
            )
            ans = res.choices[0].message.content
            active_chat["messages"].append({"role": "assistant", "content": ans})
            
            with st.chat_message("assistant"):
                st.markdown(ans, unsafe_allow_html=True)
                # Dynamic Plotly Logic
                if "plot" in ans.lower() or "graph" in ans.lower():
                    x = np.linspace(0, 10, 50)
                    y = np.sin(x) # Default example
                    fig = px.line(x=x, y=y, title="Neural Scientific Plot", template="plotly_dark")
                    st.plotly_chart(fig)
            st.rerun()
        except Exception as e:
            st.error(f"Brain Sync Error: {e}")

with tab_lab:
    st.header("🔬 3D Lab & Complex Solver")
    chem_3d = st.selectbox("Select Molecule for Deep Dive", ["None", "CH4 (Methane)", "NH3 (Ammonia)", "H2O (Water)"])
    if chem_3d != "None":
        st.subheader(f"3D Bonding Analysis: {chem_3d}")
        # Custom 3D logic here for atoms
        fig = go.Figure(data=[go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=20, color='red'))])
        st.plotly_chart(fig)