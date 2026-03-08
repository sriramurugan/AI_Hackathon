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

# 1. PAGE CONFIG & STYLING
st.set_page_config(page_title="Omni-Neural AI Tutor", layout="wide")
st.markdown("""
    <style>
    .stApp {background-color: #0e1117; color: #ffffff;}
    .stChatMessage {border-radius: 10px; margin-bottom: 10px;}
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Omni-Neural NCERT Engine")

# 2. LOAD MODELS (Ultra Logic Core)
@st.cache_resource
def load_core():
    # Embedding model for PDF Retrieval
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_core()
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# 3. INITIALIZE SESSION STATE (Memory & Knowledge)
if "messages" not in st.session_state:
    st.session_state.messages = [] # Persistent Chat History
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None

# 4. SIDEBAR: PDF ANALYSIS & CONTROL
with st.sidebar:
    st.header("📂 Knowledge Base")
    pdf = st.file_uploader("Upload NCERT PDF", type="pdf")
    
    if pdf and st.button("Analyze & Index PDF"):
        with st.spinner("Analyzing PDF content and building Neural Map..."):
            reader = PdfReader(pdf)
            # Extract and clean text for indexing
            text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
            text = re.sub(r'\s+', ' ', text)
            
            chunks = [text[i:i+600] for i in range(0, len(text), 600)]
            st.session_state.chunks = chunks
            
            # Vector Embeddings for RAG Logic
            embeddings = embed_model.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            st.session_state.index = index
            st.success("PDF Analyzed! Knowledge integrated.")

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# 5. CHAT INTERFACE (Displaying Previous Chat)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. DYNAMIC EXECUTION LOGIC
if question := st.chat_input("Ask a question or analyze data..."):
    # Add User Message to History
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # RAG Retrieval Logic
    context = ""
    if st.session_state.index is not None:
        q_embed = embed_model.encode([question])
        _, I = st.session_state.index.search(np.array(q_embed).astype('float32'), 3)
        context = "\n".join([st.session_state.chunks[i] for i in I[0]])
    else:
        context = "No PDF uploaded. Respond using general knowledge."

    # --- LLM REASONING ENGINE ---
    # System prompt provides context and language instructions
    sys_msg = {"role": "system", "content": f"Expert NCERT Tutor. Context: {context}. Speak in Tanglish."}
    
    try:
        # Full history sent to LLM for conversational memory
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[sys_msg] + st.session_state.messages
        )
        answer = response.choices[0].message.content
        
        # Add Assistant Response to History
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
            
            # --- DYNAMIC GRAPH GENERATION ENGINE ---
            # Triggered by keywords like plot, graph, or visualize
            q_low = question.lower()
            if any(word in q_low for word in ["plot", "graph", "visualize", "chart", "map"]):
                st.write("---")
                viz_prompt = f"Create Plotly Python code for a figure 'fig' based on: {question}. Return ONLY code block."
                
                viz_res = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": viz_prompt}]
                )
                
                # Execute the AI-generated plotting code
                code_match = re.search(r"```python\n(.*?)\n```", viz_res.choices[0].message.content, re.DOTALL)
                if code_match:
                    exec_env = {'np': np, 'pd': pd, 'go': go, 'px': px, 'fig': None}
                    exec(code_match.group(1), {}, exec_env)
                    if exec_env['fig']:
                        st.plotly_chart(exec_env['fig'], use_container_width=True)

    except Exception as e:
        st.error(f"Logic Engine Error: {e}")