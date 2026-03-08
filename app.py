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

# 2. ENGINES INITIALIZATION (Cached for Speed)
@st.cache_resource
def load_assets():
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return client, embedder

client, embedder = load_assets()

# 3. GLOBAL LONG-TERM MEMORY STATE
if "chats" not in st.session_state:
    st.session_state.chats = {str(uuid.uuid4()): {"title": "Main Brain", "messages": [], "db": None, "chunks": []}}
    st.session_state.active_id = list(st.session_state.chats.keys())[0]

# --- SIDEBAR: UNIVERSAL KNOWLEDGE INDEXING ---
with st.sidebar:
    st.title("🧬 Neural Archive")
    if st.button("➕ New Study Session"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Session", "messages": [], "db": None, "chunks": []}
        st.session_state.active_id = new_id
        st.rerun()

    uploaded_files = st.file_uploader("📂 Upload Any NCERT / Data", type=["pdf", "png", "jpg"], accept_multiple_files=True)
    if st.button("🏗️ Index Knowledge") and uploaded_files:
        with st.spinner("Analyzing Deeply..."):
            raw_text = ""
            for f in uploaded_files:
                if f.type == "application/pdf":
                    pdf = PyPDF2.PdfReader(f)
                    raw_text += "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
                else:
                    raw_text += pytesseract.image_to_string(Image.open(f))
            
            # Universal NLP Chunking & FAISS Vector Store
            chunks = [raw_text[i:i+800] for i in range(0, len(raw_text), 800)]
            if chunks:
                embeddings = embedder.encode(chunks)
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(np.array(embeddings).astype('float32'))
                st.session_state.chats[st.session_state.active_id]["db"] = index
                st.session_state.chats[st.session_state.active_id]["chunks"] = chunks
                st.success("Knowledge Synced Pakkava!")

# --- MAIN INTERFACE: THE UNIVERSAL CONSOLE ---
aid = st.session_state.active_id
active_chat = st.session_state.chats[aid]

st.title("💬 Omni-NCERT Universal AI Tutor")
st.markdown("Ask anything from **6th to 12th Std (Math, Physics, Chem)** or request **ANY NLP/Data Graph**. I will solve it step-by-step and code the visuals dynamically!")

# Render Long-Term Memory
for m in active_chat["messages"]:
    with st.chat_message(m["role"]): st.markdown(m["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Enter your complex question or visualization request machi..."):
    active_chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # Universal RAG Context Retrieval
    context = ""
    if active_chat["db"]:
        q_emb = embedder.encode([prompt])
        _, I = active_chat["db"].search(np.array(q_emb).astype('float32'), k=5)
        context = "\n".join([active_chat["chunks"][i] for i in I[0]])

    # 🧠 THE "ZERO-HARDCODE" MASTER PROMPT
    sys_msg = f"""You are the Ultimate NCERT AI Tutor for 6th to 12th standard (Maths, Physics, Chemistry, and NLP Data).
    Context from uploaded files (if any): {context}
    
    CRITICAL RULES:
    1. TONE: Speak freely in friendly Tanglish (mix of Tamil and English). Use words like 'Machi', 'Vada', 'Puriyudha', 'Gavanichiko'.
    2. PEDAGOGY: WHATEVER the difficulty (6th grade fractions to 12th grade calculus/quantum mechanics), you MUST explain STEP-BY-STEP. Do NOT just give the final answer. Act like a patient teacher.
    3. FORMULAS: Highlight any equations by starting with <div class='formula-box'>📌 **KEY FORMULA:** [LaTeX]</div>.
    4. UNIVERSAL GRAPHING ENGINE (NO HARDCODING):
       - If the user asks for ANY visualization (e.g., math function, physics trajectory, chemistry 3D molecule, or NLP bar/pie charts for text data), you MUST write Python code to generate it.
       - Use `plotly.graph_objects as go` or `plotly.express as px`.
       - You can use `numpy as np` and `pandas as pd`.
       - Assign the final plotly figure to a variable strictly named `fig`.
       - Enclose ONLY the python code in a ```python ... ``` block. Do not use plt.show().
       - Make the charts look highly professional with `template="plotly_dark"`.
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
            
            # 🚀 THE UNIVERSAL DYNAMIC EXECUTION ENGINE 🚀
            # Extracts LLM-generated code and runs ANY graph requested!
            code_match = re.search(r'```python\n(.*?)\n```', ans, re.DOTALL)
            if code_match:
                st.info("⚙️ Machi, executing dynamic graphing engine...")
                code_str = code_match.group(1)
                
                # Highly secure and capable execution environment
                local_vars = {'np': np, 'go': go, 'px': px, 'sp': sp, 'pd': pd, 'fig': None}
                try:
                    exec(code_str, globals(), local_vars)
                    if local_vars.get('fig') is not None:
                        st.plotly_chart(local_vars['fig'], use_container_width=True)
                    else:
                        st.warning("Graph generated, but variable 'fig' was not assigned by the AI.")
                except Exception as exec_err:
                    st.error(f"Graph Code Error: {exec_err}. (Tip: Ask the AI to fix the python code!)")

        st.rerun()
    except Exception as e:
        st.error(f"Brain Sync Error: {e}")