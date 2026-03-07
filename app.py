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
import plotly.graph_objects as go
import plotly.express as px

# 1. ULTIMATE UI CONFIGURATION
st.set_page_config(page_title="Omni-NCERT Neural Engine", page_icon="🎓", layout="wide")

# High-Tech Cyber-Lab CSS
st.markdown("""
    <style>
    .stApp { background: #05070a; color: #e0e6ed; font-family: 'Inter', sans-serif; }
    h1 { background: linear-gradient(90deg, #00ffcc, #0088ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem !important; font-weight: 800; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0d1117; border-radius: 15px; padding: 5px; border: 1px solid #1f2937; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #00ffcc !important; background: rgba(0, 255, 204, 0.1); border-radius: 10px; }
    section[data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 1px solid #1f2937; }
    .formula-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. CORE BRAIN INITIALIZATION
if "GROQ_API_KEY" not in st.secrets:
    st.error("🔑 Critical Error: GROQ_API_KEY not found in Streamlit Secrets.")
    st.stop()

@st.cache_resource
def load_system_engines():
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return client, embedder

client, embedder = load_system_engines()

# 3. STATE MANAGEMENT
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_id" not in st.session_state:
    st.session_state.active_id = str(uuid.uuid4())
    st.session_state.chats[st.session_state.active_id] = {"messages": [], "title": "Main Session", "db": None, "chunks": []}

# 4. SIDEBAR: KNOWLEDGE MANAGEMENT
with st.sidebar:
    st.title("🏛️ Neural Archive")
    if st.button("➕ Create New Study Session", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"messages": [], "title": "New Session", "db": None, "chunks": []}
        st.session_state.active_id = new_id
        st.rerun()

    st.divider()
    st.subheader("📁 Textbook Indexer")
    uploads = st.file_uploader("Upload NCERT PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("🧠 Deep-Index Books") and uploads:
        with st.spinner("Processing NCERT Data..."):
            text_data = ""
            for f in uploads:
                pdf = PyPDF2.PdfReader(f)
                text_data += "\n".join([page.extract_text() for page in pdf.pages])
            
            chunks = [text_data[i:i+700] for i in range(0, len(text_data), 700)]
            embeddings = embedder.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            
            curr = st.session_state.chats[st.session_state.active_id]
            curr["db"], curr["chunks"] = index, chunks
            st.success("Indexing Complete!")

# --- MAIN INTERFACE ---
st.title("Omni-NCERT Neural Engine")

tab_chat, tab_math, tab_graphs, tab_formulas = st.tabs([
    "🧠 Senior AI Tutor", "📐 Symbolic Math Solver", "📊 Global Graph Lab", "📜 Formula Vault"
])

# TAB 1: AI TUTOR (RAG)
with tab_chat:
    active_chat = st.session_state.chats[st.session_state.active_id]
    for m in active_chat["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask any NCERT question (Class 6-12)..."):
        active_chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # Context retrieval
        context = ""
        if active_chat["db"]:
            q_emb = embedder.encode([prompt])
            _, I = active_chat["db"].search(np.array(q_emb).astype('float32'), k=4)
            context = "\n".join([active_chat["chunks"][i] for i in I[0]])

        system_msg = f"""You are the Omni-NCERT Senior Tutor. 
        SYLLABUS: Class 6 to 12. SUBJECTS: Math, Physics, Chemistry.
        CONTEXT: {context}
        RULES: 
        1. Give step-by-step derivations.
        2. Use LaTeX for all math: $E=mc^2$.
        3. If context is missing, use internal NCERT knowledge."""

        try:
            res = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[{"role": "system", "content": system_msg}] + active_chat["messages"],
                temperature=0.1
            )
            ans = res.choices[0].message.content
            active_chat["messages"].append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"): st.markdown(ans)
        except Exception as e:
            st.error(f"Brain Sync Error: {e}")

# TAB 2: SYMBOLIC MATH (Solving Complex Equations)
with tab_math:
    st.header("📐 Precise Symbolic Solver")
    eq_input = st.text_input("Enter Equation (e.g., x**2 + 5*x + 6)", "x**2 + 5*x + 6")
    if st.button("Solve Step-by-Step"):
        x = sympy.symbols('x')
        try:
            expr = sympy.sympify(eq_input)
            sol = sympy.solve(expr, x)
            st.latex(rf"\text{{Factors: }} {sympy.factor(expr)}")
            st.latex(rf"\text{{Solutions: }} {sol}")
            st.latex(rf"\text{{Derivative: }} {sympy.diff(expr, x)}")
            st.latex(rf"\text{{Integral: }} {sympy.integrate(expr, x)}")
        except:
            st.error("Please enter a valid algebraic expression.")

# TAB 3: GLOBAL GRAPH LAB
with tab_graphs:
    st.header("📊 Multi-Subject Graphing Engine")
    g_type = st.selectbox("Topic", ["Kinematics", "Wave Optics", "Chemical Kinetics", "Trigonometry"])
    
    fig = go.Figure()
    if g_type == "Kinematics":
        u = st.slider("u (m/s)", 0, 100, 20); a = st.slider("a (m/s²)", -10, 10, 2)
        t = np.linspace(0, 10, 100); s = u*t + 0.5*a*t**2
        fig.add_trace(go.Scatter(x=t, y=s, mode='lines', name='Position-Time', line=dict(color='#00ffcc')))
        fig.update_layout(title="Displacement-Time Graph", xaxis_title="Time (s)", yaxis_title="Displacement (m)")
    elif g_type == "Wave Optics":
        wl = st.slider("Wavelength (nm)", 400, 700, 500)
        x = np.linspace(-1, 1, 1000); y = np.cos(wl * x)**2
        fig = px.line(x=x, y=y, title="Interference Pattern", color_discrete_sequence=['#0088ff'])
    
    st.plotly_chart(fig, use_container_width=True)

# TAB 4: FORMULA VAULT
with tab_formulas:
    st.header("📜 NCERT Chapter-wise Formulas")
    grade = st.selectbox("Select Class", [10, 11, 12])
    sub = st.radio("Subject", ["Physics", "Chemistry", "Math"], horizontal=True)
    
    formulas = {
        (12, "Physics"): {"Electrostatics": r"F = \frac{1}{4\pi\epsilon_0}\frac{q_1q_2}{r^2}", "Gauss Law": r"\oint E \cdot dA = \frac{q}{\epsilon_0}"},
        (11, "Physics"): {"Work-Energy": r"W = \Delta K", "Gravitation": r"F = G\frac{m_1m_2}{r^2}"}
    }
    
    current_f = formulas.get((grade, sub), {"General": "Formulas loading..."})
    for chapter, f_text in current_f.items():
        st.markdown(f"""<div class='formula-card'><h4>{chapter}</h4></div>""", unsafe_allow_html=True)
        st.latex(f_text)