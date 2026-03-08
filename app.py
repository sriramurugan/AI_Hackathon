import streamlit as st
import numpy as np
import faiss
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from groq import Groq
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# 1. PAGE CONFIG (Hackathon Ready)
st.set_page_config(page_title="Omni-Neural AI Tutor", layout="wide")
st.title("🧬 Omni-Neural NCERT Engine")

# 2. LOAD MODELS (Ultra-Fast Cache)
@st.cache_resource
def load_core():
    # MiniLM chosen for speed (<2s embedding)
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_core()
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# 3. SESSION STATE (Chat Memory & Index)
if "messages" not in st.session_state: st.session_state.messages = []
if "chunks" not in st.session_state: st.session_state.chunks = []
if "index" not in st.session_state: st.session_state.index = None

# 4. SIDEBAR: INSTANT PDF ANALYSIS (< 2 Seconds Logic)
with st.sidebar:
    st.header("📂 Knowledge Base")
    pdf = st.file_uploader("Upload NCERT PDF", type="pdf")
    
    if pdf and st.button("⚡ Instant Index"):
        with st.spinner("Neural Mapping..."):
            reader = PdfReader(pdf)
            # SPEED FIX: Analyze first 40-50 pages for immediate context
            pages_to_read = reader.pages[:50]
            text = " ".join([p.extract_text() for p in pages_to_read if p.extract_text()])
            
            # Optimized chunk size for faster vector search
            chunks = [text[i:i+800] for i in range(0, len(text), 800)]
            st.session_state.chunks = chunks
            
            # Fast batch vectorization
            embeddings = embed_model.encode(chunks, batch_size=32, show_progress_bar=False)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            st.session_state.index = index
            st.success("Indexing Complete! (~2s)")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# 5. CHAT HISTORY UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. ULTRA-STRONG REASONING & GRAPH ENGINE
if question := st.chat_input("Ask about PDF, Maths, or Algorithms..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieval Logic
    context = ""
    if st.session_state.index is not None:
        q_embed = embed_model.encode([question])
        _, I = st.session_state.index.search(np.array(q_embed).astype('float32'), 3)
        context = "\n".join([st.session_state.chunks[i] for i in I[0]])

    try:
        # LLM Reasoning with History
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": f"Expert NCERT Tutor. Context: {context}. Speak in Tanglish."}] + st.session_state.messages
        )
        answer = res.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        with st.chat_message("assistant"):
            st.markdown(answer)
            
            # --- ULTRA-STRONG GRAPH ENGINE (Maths & Algo) ---
            q_low = question.lower()
            if any(word in q_low for word in ["plot", "graph", "chart", "linear", "algo", "map"]):
                st.write("---")
                viz_prompt = f"""
                Objective: Plotly figure 'fig' for: {question}
                RULES:
                1. Use pandas: df = pd.DataFrame(...)
                2. Use plotly.express as px
                3. Maths: Use np.linspace for smooth curves.
                4. Algo: Create dummy data clusters/points.
                5. NO 'return'. Just create 'fig'. Wrap range() in list().
                Return ONLY ```python block.
                """
                
                viz_res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": viz_prompt}])
                code_match = re.search(r"```python\n(.*?)\n```", viz_res.choices[0].message.content, re.DOTALL)
                
                if code_match:
                    exec_env = {'np': np, 'pd': pd, 'go': go, 'px': px, 'fig': None}
                    exec(code_match.group(1), {}, exec_env)
                    if exec_env['fig']:
                        st.plotly_chart(exec_env['fig'], use_container_width=True)

    except Exception as e:
        st.error(f"Logic Engine Error: {e}")