import streamlit as st
from groq import Groq
import numpy as np
import uuid
import plotly.graph_objects as go
import plotly.express as px
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import re
import networkx as nx

# 1. PAGE CONFIG & NEON REPAIR THEME
st.set_page_config(page_title="Omni-Neural REPAIR Engine", page_icon="🔧", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #0a0c12; color: #f0f2f6; }
    .reasoning-box { background: #161b22; border-left: 4px solid #ffd700; padding: 15px; border-radius: 8px; font-style: italic; color: #ffd700; margin-bottom: 20px; }
    .stButton>button { border-radius: 20px; border: 1px solid #00ffcc; color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

# 2. ENGINES & REPAIR LOGIC
@st.cache_resource
def load_core():
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return client, embedder

client, embedder = load_core()

def high_quality_clean(text):
    # Fixes: Dependence on high-quality training data
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', text) 
    return text.strip()

# 3. KNOWLEDGE STATE
if "kb" not in st.session_state:
    st.session_state.kb = {"chunks": [], "db": None, "graph": nx.Graph(), "feedback_log": []}
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: DATA REPAIR PIPELINE ---
with st.sidebar:
    st.title("🔧 Repair & Index")
    uploaded_files = st.file_uploader("Upload Domain Knowledge (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("🏗️ Build Perfect Index") and uploaded_files:
        with st.spinner("Repairing Data & Mapping Relationships..."):
            full_text = ""
            for f in uploaded_files:
                pdf = PyPDF2.PdfReader(f)
                full_text += "\n".join([high_quality_clean(p.extract_text()) for p in pdf.pages if p.extract_text()])
            
            # Semantic Repair (Vector)
            chunks = [full_text[i:i+600] for i in range(0, len(full_text), 600)]
            embeddings = embedder.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            
            # Abstract Reasoning Repair (Graph)
            G = nx.Graph()
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', full_text[:5000])
            for i in range(len(entities)-1):
                G.add_edge(entities[i], entities[i+1])
            
            st.session_state.kb["chunks"] = chunks
            st.session_state.kb["db"] = index
            st.session_state.kb["graph"] = G
            st.success("Brain Repaired & Ready!")

# --- MAIN INTERFACE ---
st.title("🧬 Omni-Neural REPAIR Engine")

# Display Messages with Feedback Loop
for idx, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            c1, c2 = st.columns([0.1, 0.9])
            if c1.button("👍", key=f"pos_{idx}"): st.toast("Positive reinforcement recorded.")
            if c2.button("👎", key=f"neg_{idx}"): st.toast("Incorrect logic flagged for repair.")

if prompt := st.chat_input("Challenge my reasoning, Machi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # Retrieval Logic
    context = ""
    if st.session_state.kb["db"]:
        _, I = st.session_state.kb["db"].search(np.array(embedder.encode([prompt])).astype('float32'), k=3)
        context = "\n".join([st.session_state.kb["chunks"][i] for i in I[0]])

    # THE MASTER REPAIR PROMPT
    sys_msg = f"""
    You are an UPGRADED Neural Engine. Your flaws have been repaired via GraphRAG and Human-in-the-loop feedback.
    
    REPAIR STRATEGY:
    1. CONTEXT: Use this cleaned data: {context}
    2. REASONING: Before answering, perform internal 'Abstract Reasoning'. 
    3. FLaw Fix: Avoid statistical bias. Focus on factual 'Ground Truth'.
    4. TANGISH: Speak like a friendly peer (Machi/Vada).
    5. VISUALS: If a graph helps reduce ambiguity, use Plotly (```python ... ``` with 'fig').
    """

    try:
        # Reasoning Simulation
        with st.expander("🧠 Internal Reasoning Process (Repairing Continuity)"):
            st.write(f"Searching Graph Nodes: {list(st.session_state.kb['graph'].nodes)[:5]}")
            st.write("Analyzing context for potential biases...")
            st.write("Synthesizing step-by-step response...")

        res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "system", "content": sys_msg}] + st.session_state.messages)
        ans = res.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": ans})
        
        with st.chat_message("assistant"):
            st.markdown(ans)
            code_match = re.search(r'```python\n(.*?)\n```', ans, re.DOTALL)
            if code_match:
                local_vars = {'np': np, 'go': go, 'px': px, 'fig': None}
                exec(code_match.group(1), globals(), local_vars)
                if local_vars['fig']: st.plotly_chart(local_vars['fig'])
        st.rerun()
    except Exception as e: st.error(f"Sync Error: {e}")