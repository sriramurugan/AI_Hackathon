import streamlit as st
from groq import Groq
import numpy as np
import pandas as pd
import uuid
import urllib.parse
import plotly.graph_objects as go
from PIL import Image
import pytesseract
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss

# 1. PAGE CONFIG & NEON THEME
st.set_page_config(page_title="Omni-NCERT Neural Engine", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #05070a; color: #e0e6ed; }
    h1 { background: linear-gradient(90deg, #00ffcc, #0088ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    .stButton>button { background: #161b22; border: 1px solid #00ffcc; color: #00ffcc; border-radius: 8px; width: 100%; }
    .stChatMessage { border: 1px solid #1f2937; border-radius: 15px; margin-bottom: 10px; background: #0d1117; }
    .formula-box { 
        background: linear-gradient(90deg, #1f2937, #111827); 
        border-left: 5px solid #00ffcc; 
        padding: 15px; border-radius: 10px; margin-bottom: 20px;
        color: #00ffcc; font-family: 'Courier New', Courier, monospace;
    }
    .action-link { color: #00ffcc !important; text-decoration: none !important; border: 1px solid #00ffcc; padding: 4px 10px; border-radius: 5px; font-size: 0.8rem; }
    section[data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 1px solid #1f2937; }
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
    st.session_state.chats = {}
if "active_id" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chats[first_id] = {"title": "Main Brain", "messages": [], "db": None, "chunks": []}
    st.session_state.active_id = first_id

# --- SIDEBAR: NAVIGATION ---
with st.sidebar:
    st.title("🧬 Neural Archive")
    if st.button("➕ New Study Session"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Session", "messages": [], "db": None, "chunks": []}
        st.session_state.active_id = new_id
        st.rerun()

    st.divider()
    uploaded_files = st.file_uploader("📂 Upload NCERT (PDF/IMG)", type=["pdf", "png", "jpg"], accept_multiple_files=True)
    if st.button("🏗️ Index Knowledge") and uploaded_files:
        with st.spinner("Analyzing..."):
            raw_text = ""
            for f in uploaded_files:
                if f.type == "application/pdf":
                    pdf = PyPDF2.PdfReader(f)
                    raw_text += "\n".join([p.extract_text() for p in pdf.pages])
                else:
                    raw_text += pytesseract.image_to_string(Image.open(f))
            chunks = [raw_text[i:i+600] for i in range(0, len(raw_text), 600)]
            embeddings = embedder.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            st.session_state.chats[st.session_state.active_id]["db"] = index
            st.session_state.chats[st.session_state.active_id]["chunks"] = chunks
            st.success("Indexed!")

# --- MAIN INTERFACE ---
aid = st.session_state.active_id
active_chat = st.session_state.chats[aid]

tab_chat, tab_lab = st.tabs(["💬 AI Tutor Console", "📊 Pro Visual Lab"])

with tab_chat:
    for idx, m in enumerate(active_chat["messages"]):
        with st.chat_message(m["role"]):
            st.markdown(m["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Physics/Maths formula kelu, illa Chem 3D kelu machi..."):
        active_chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        context = ""
        if active_chat["db"]:
            q_emb = embedder.encode([prompt])
            _, I = active_chat["db"].search(np.array(q_emb).astype('float32'), k=3)
            context = "\n".join([active_chat["chunks"][i] for i in I[0]])

        # 🧠 THE "PAKKAVA" TANGGLISH PROMPT
        sys_msg = f"""You are a Pro NCERT Science Teacher. Context: {context}.
        RULES:
        1. Speak only in Tanglish (Tamil + English mixture). Use friendly words like 'Machi', 'Vada', 'Puriyudha'.
        2. Physics/Maths: START response with <div class='formula-box'>📌 **KEY FORMULAS:** [Insert LaTeX formulas here]</div>.
        3. Chemistry: Give deep explanations on hybridization and 3D bonding.
        4. Model: Use llama-3.1-8b-instant for zero lag and high speed."""

        try:
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": sys_msg}] + active_chat["messages"]
            )
            ans = res.choices[0].message.content
            active_chat["messages"].append({"role": "assistant", "content": ans})
            st.rerun()
        except Exception as e:
            st.error(f"Brain Sync Error: {e}")

with tab_lab:
    st.header("🔬 High-Fidelity 3D Lab")
    sim = st.selectbox("Simulation", ["Methane 3D Structure", "Projectile Motion", "Ohm's Law"])
    fig = go.Figure()

    if sim == "Methane 3D Structure":
        # CH4 Tetrahedral
        x = [0, 1, -1, -1, 1]; y = [0, 1, 1, -1, -1]; z = [0, 1, -1, 1, -1]
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers+text', 
                                   marker=dict(size=[20, 12, 12, 12, 12], color=['black', 'white', 'white', 'white', 'white'])))
        fig.update_layout(title="Methane (CH4) Tetrahedral Shape", template="plotly_dark")
    
    # ... (Other simulation code)
    st.plotly_chart(fig, use_container_width=True)