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
    .action-link { color: #00ffcc !important; text-decoration: none !important; border: 1px solid #00ffcc; padding: 4px 10px; border-radius: 5px; font-size: 0.8rem; }
    section[data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 1px solid #1f2937; }
    </style>
    """, unsafe_allow_html=True)

# 2. ENGINES INITIALIZATION
if "GROQ_API_KEY" not in st.secrets:
    st.error("🔑 GROQ_API_KEY missing in Secrets!")
    st.stop()

@st.cache_resource
def load_assets():
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return client, embedder

client, embedder = load_assets()

# 3. GLOBAL STATE: MULTI-CHAT HISTORY & DEFENSIVE LOADING
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "active_id" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chats[first_id] = {"title": "Main Brain", "messages": [], "db": None, "chunks": []}
    st.session_state.active_id = first_id

# Fix for potential KeyErrors during version updates
for cid in st.session_state.chats:
    if "title" not in st.session_state.chats[cid]: st.session_state.chats[cid]["title"] = "Study Session"

# --- SIDEBAR: NAVIGATION & INDEXING ---
with st.sidebar:
    st.title("🧬 Neural Archive")
    
    if st.button("➕ New Study Session", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Session", "messages": [], "db": None, "chunks": []}
        st.session_state.active_id = new_id
        st.rerun()

    st.divider()
    st.subheader("📚 History")
    for chat_id, chat_data in st.session_state.chats.items():
        if st.button(f"🗨️ {chat_data['title'][:20]}", key=chat_id, use_container_width=True):
            st.session_state.active_id = chat_id
            st.rerun()

    st.divider()
    uploaded_files = st.file_uploader("📂 Upload NCERT (PDF/IMG)", type=["pdf", "png", "jpg"], accept_multiple_files=True)
    if st.button("🏗️ Index Knowledge") and uploaded_files:
        with st.spinner("Analyzing Content..."):
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
            st.success("Indexed Successfully!")

# --- MAIN INTERFACE ---
aid = st.session_state.active_id
active_chat = st.session_state.chats[aid]

st.title("Omni-NCERT Neural Engine")

tab_chat, tab_lab = st.tabs(["💬 AI Tutor Console", "📊 Pro Visual Lab"])

with tab_chat:
    # Display messages
    for idx, m in enumerate(active_chat["messages"]):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                wa_msg = urllib.parse.quote(f"NCERT Solution:\n{m['content'][:500]}")
                wa_url = f"https://wa.me/?text={wa_msg}"
                cols = st.columns([0.15, 0.15, 0.2, 0.5])
                cols[0].button("👍", key=f"l_{idx}_{aid}")
                cols[1].button("👎", key=f"d_{idx}_{aid}")
                cols[2].markdown(f'<a href="{wa_url}" target="_blank" class="action-link">📲 Share</a>', unsafe_allow_html=True)

    if prompt := st.chat_input("Ask any Class 6-12 question..."):
        if not active_chat["messages"]: active_chat["title"] = prompt[:25]
        active_chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # RAG Logic
        context = ""
        if active_chat["db"]:
            q_emb = embedder.encode([prompt])
            _, I = active_chat["db"].search(np.array(q_emb).astype('float32'), k=3)
            context = "\n".join([active_chat["chunks"][i] for i in I[0]])

        try:
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": f"You are a Senior NCERT Tutor. Context: {context}. Use LaTeX $$."}] + active_chat["messages"],
                temperature=0.2
            )
            ans = res.choices[0].message.content
            active_chat["messages"].append({"role": "assistant", "content": ans})
            st.rerun()
        except Exception as e:
            st.error(f"Brain Sync Error: {e}")

with tab_lab:
    st.header("🔬 High-Fidelity Neural Lab")
    sim = st.selectbox("Simulation", ["Projectile Motion", "Ice Cream Sales (Linear)", "Ohm's Law"])
    fig = go.Figure()

    if sim == "Projectile Motion":
        v0 = st.slider("u (m/s)", 10, 100, 50); angle = st.slider("Angle (°)", 10, 80, 45)
        t_f = (2 * v0 * np.sin(np.radians(angle))) / 9.8
        t = np.linspace(0, t_f, 100)
        x = v0 * t * np.cos(np.radians(angle)); y = v0 * t * np.sin(np.radians(angle)) - 0.5 * 9.8 * t**2
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#00ffcc', width=4)))
        fig.update_layout(title="Trajectory Path", xaxis_title="Range (m)", yaxis_title="Height (m)", template="plotly_dark")

    elif sim == "Ice Cream Sales (Linear)":
        m = st.number_input("Slope (m)", value=15); c = st.number_input("Intercept (c)", value=50)
        temp = np.linspace(10, 50, 100); sales = m * temp + c
        fig.add_trace(go.Scatter(x=temp, y=sales, mode='lines+markers', line=dict(color='#0088ff')))
        fig.update_layout(title="Linear Equation: y = mx + c", xaxis_title="Temp (°C)", yaxis_title="Sales ($)", template="plotly_dark")

    elif sim == "Ohm's Law":
        r = st.select_slider("R (Ω)", options=[10, 20, 50, 100])
        v = np.linspace(0, 12, 50); i = v / r
        fig.add_trace(go.Scatter(x=v, y=i, mode='lines', fill='tozeroy', line=dict(color='#ff0055')))
        fig.update_layout(title="V-I Characteristics", xaxis_title="Voltage (V)", yaxis_title="Current (I)", template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)