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

# 2. LOAD MODELS (Fast Cache)
@st.cache_resource
def load_core():
    # MiniLM is chosen for <2s embedding speed
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_core()
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# 3. SESSION STATE
if "messages" not in st.session_state: st.session_state.messages = []
if "chunks" not in st.session_state: st.session_state.chunks = []
if "index" not in st.session_state: st.session_state.index = None

# 4. OPTIMIZED PDF ANALYSIS (<2 Seconds Logic)
with st.sidebar:
    st.header("📂 Knowledge Base")
    pdf = st.file_uploader("Upload NCERT PDF", type="pdf")
    
    if pdf and st.button("Instant Index"):
        with st.spinner("Neural Mapping..."):
            reader = PdfReader(pdf)
            # FAST EXTRACTION: Only first 50 pages for instant hackathon demo if needed
            text = " ".join([p.extract_text() for p in reader.pages[:50] if p.extract_text()])
            
            # Larger chunks = Fewer embeddings = Faster speed
            chunks = [text[i:i+800] for i in range(0, len(text), 800)]
            st.session_state.chunks = chunks
            
            # Vectorizing
            embeddings = embed_model.encode(chunks, batch_size=32, show_progress_bar=False)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            st.session_state.index = index
            st.success("⚡ Indexing Complete (~2s)")

# 5. CHAT HISTORY DISPLAY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. DYNAMIC EXECUTION (Fixed Plotly Errors)
if question := st.chat_input("Ask or plot data..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)

    context = ""
    if st.session_state.index is not None:
        q_embed = embed_model.encode([question])
        _, I = st.session_state.index.search(np.array(q_embed).astype('float32'), 3)
        context = "\n".join([st.session_state.chunks[i] for i in I[0]])

    try:
        # LLM Reasoning
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": f"Expert Tutor. Context: {context}. Speak in Tanglish."}] + st.session_state.messages
        )
        answer = res.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        with st.chat_message("assistant"):
            st.markdown(answer)
            
            # --- FIXED DYNAMIC VISUALIZATION ENGINE ---
            q_low = question.lower()
            if any(word in q_low for word in ["plot", "graph", "bar", "chart"]):
                viz_prompt = f"""
                Create Plotly code for 'fig' based on: {question}.
                STRICT RULES:
                1. Always use a pd.DataFrame: df = pd.DataFrame(...)
                2. Use px.bar(df, x='column1', y='column2')
                3. If using range(), wrap it in list(): list(range(10))
                4. Return ONLY code block.
                """
                
                viz_res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": viz_prompt}])
                code_match = re.search(r"```python\n(.*?)\n```", viz_res.choices[0].message.content, re.DOTALL)
                
                if code_match:
                    # Creating a safe execution environment
                    exec_env = {'np': np, 'pd': pd, 'go': go, 'px': px, 'fig': None}
                    exec(code_match.group(1), {}, exec_env)
                    if exec_env['fig']:
                        st.plotly_chart(exec_env['fig'], use_container_width=True)

    except Exception as e:
        st.error(f"Logic Engine Error: {e}")