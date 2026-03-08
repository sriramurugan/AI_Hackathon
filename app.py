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

# 1. PAGE CONFIG
st.set_page_config(page_title="Omni-Neural AI Tutor", layout="wide")
st.title("🧬 Omni-Neural NCERT Engine")

# 2. LOAD MODELS
@st.cache_resource
def load_core():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_core()
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# 3. SESSION STATE
if "chunks" not in st.session_state: st.session_state.chunks = []
if "index" not in st.session_state: st.session_state.index = None

# 4. SIDEBAR: PDF UPLOAD & INDEXING
with st.sidebar:
    st.header("📂 Knowledge Base")
    pdf = st.file_uploader("Upload NCERT PDF", type="pdf")
    if pdf and st.button("Build Neural Index"):
        with st.spinner("Processing..."):
            reader = PdfReader(pdf)
            text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
            chunks = [text[i:i+600] for i in range(0, len(text), 600)]
            st.session_state.chunks = chunks
            
            embeddings = embed_model.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            st.session_state.index = index
            st.success("Brain Updated!")

# 5. CHAT INTERFACE
question = st.chat_input("Ask about Physics, Chemistry, Maths or Algorithms...")

if question:
    st.write(f"### ❓ Question: {question}")
    
    # RAG Retrieval
    context = ""
    if st.session_state.index is not None:
        q_embed = embed_model.encode([question])
        _, I = st.session_state.index.search(np.array(q_embed).astype('float32'), 3)
        context = "\n".join([st.session_state.chunks[i] for i in I[0]])
    
    # --- TEXT RESPONSE ENGINE ---
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nExplain step-by-step with formulas in Tanglish."
    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        st.write("### 🤖 Answer")
        st.write(res.choices[0].message.content)
    except Exception as e:
        st.error(f"API Error: {e}")

    # --- DYNAMIC GRAPH ENGINE (The "Super" Fix) ---
    q_lower = question.lower()
    if any(word in q_lower for word in ["plot", "graph", "visualize", "chart", "map"]):
        st.write("---")
        st.write("### 📊 Neural Visualization")
        
        viz_prompt = f"""
        Based on: {question}
        Generate Python code to create a Plotly figure named 'fig'.
        Support: Bar, Heatmap, Scatter, Pie, Histogram, Box plot, or Math Functions.
        If data is needed, create a small sample dictionary/dataframe.
        Return ONLY the code block starting with ```python and ending with ```.
        Do not include st.plotly_chart(fig) inside the block.
        """
        
        try:
            viz_res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": viz_prompt}]
            )
            code_out = viz_res.choices[0].message.content
            clean_code = re.search(r"```python\n(.*?)\n```", code_out, re.DOTALL)
            
            if clean_code:
                exec_env = {'np': np, 'pd': pd, 'go': go, 'px': px, 'fig': None}
                exec(clean_code.group(1), {}, exec_env)
                if exec_env['fig']:
                    st.plotly_chart(exec_env['fig'], use_container_width=True)
            else:
                st.warning("AI couldn't generate the specific graph code.")
        except Exception as e:
            st.error(f"Visual Engine Error: {e}")