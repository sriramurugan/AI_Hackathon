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

# 1. PAGE CONFIG & NEON THEME
st.set_page_config(page_title="Omni-NCERT Neural Engine", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #05070a; color: #e0e6ed; }
    h1 { background: linear-gradient(90deg, #00ffcc, #0088ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    .stButton>button { background: #161b22; border: 1px solid #00ffcc; color: #00ffcc; border-radius: 8px; width: 100%; }
    .stChatMessage { border: 1px solid #1f2937; border-radius: 15px; margin-bottom: 10px; }
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

# 3. SESSION STATE (History & RAG)
if "chats" not in st.session_state:
    st.session_state.chats = {str(uuid.uuid4()): {"messages": [], "db": None, "chunks": []}}
    st.session_state.active_id = list(st.session_state.chats.keys())[0]

# --- SIDEBAR: KNOWLEDGE INDEX & HISTORY ---
with st.sidebar:
    st.title("🧬 Neural Archive")
    
    # PDF/Image Uploader
    uploaded_files = st.file_uploader("📂 Upload NCERT Docs", type=["pdf", "png", "jpg"], accept_multiple_files=True)
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
            st.success("Indexing Complete!")

    st.divider()
    
    # Download Chat History
    history_text = ""
    for m in st.session_state.chats[st.session_state.active_id]["messages"]:
        history_text += f"{m['role'].upper()}: {m['content']}\n\n"
    
    st.download_button(
        label="💾 Download This Session",
        data=history_text,
        file_name=f"ncert_session_{st.session_state.active_id[:8]}.txt",
        mime="text/plain"
    )

# --- MAIN INTERFACE ---
st.title("Omni-NCERT Neural Engine")

tab_chat, tab_lab = st.tabs(["💬 AI Tutor Console", "📊 Visual Lab"])

with tab_chat:
    active_chat = st.session_state.chats[st.session_state.active_id]
    
    for m in active_chat["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask about Math, Physics, or Chemistry..."):
        active_chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG Search
        context = ""
        if active_chat["db"]:
            q_emb = embedder.encode([prompt])
            _, I = active_chat["db"].search(np.array(q_emb).astype('float32'), k=3)
            context = "\n".join([active_chat["chunks"][i] for i in I[0]])

        # Llama 3.3-70B reasoning
        sys_msg = f"You are a Senior NCERT Tutor. Context: {context}. Use LaTeX $$ for math and chemical formulas."
        try:
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": sys_msg}] + active_chat["messages"],
                temperature=0.2
            )
            ans = res.choices[0].message.content
            active_chat["messages"].append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"):
                st.markdown(ans)
        except Exception as e:
            st.error(f"Error: {e}")

with tab_lab:
    st.header("📉 Physics & Math Simulator")
    sim = st.selectbox("Experiment", ["Projectile Motion", "Ice Cream Sales (Linear)"])
    
    if sim == "Projectile Motion":
        v = st.slider("Velocity", 10, 100, 50)
        a = st.slider("Angle", 10, 80, 45)
        t_m = (2 * v * np.sin(np.radians(a))) / 9.8
        tr = np.linspace(0, t_m, 100)
        x = v * tr * np.cos(np.radians(a)); y = v * tr * np.sin(np.radians(a)) - 0.5 * 9.8 * tr**2
        st.line_chart(pd.DataFrame({"Range (m)": x, "Height (m)": y}), x="Range (m)", y="Height (m)")
        
    elif sim == "Ice Cream Sales (Linear)":
        temp = np.linspace(15, 45, 100)
        sales = 15 * temp + 50
        st.line_chart(pd.DataFrame({"Temp (°C)": temp, "Sales ($)": sales}), x="Temp (°C)", y="Sales ($)")