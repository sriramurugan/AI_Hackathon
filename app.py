import streamlit as st
from groq import Groq
import numpy as np
import pandas as pd
import uuid
import urllib.parse
from PIL import Image
import pytesseract
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss

# 1. PAGE CONFIG & ENHANCED CSS
st.set_page_config(page_title="Omni-NCERT Neural Engine", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #05070a; color: #e0e6ed; }
    h1 { background: linear-gradient(90deg, #00ffcc, #0088ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    
    /* Sidebar Session Cards */
    .session-card {
        padding: 10px; background: #161b22; border-radius: 8px; 
        border: 1px solid #30363d; margin-bottom: 10px; cursor: pointer;
    }
    
    /* Action Buttons Styling */
    .chat-actions { display: flex; gap: 10px; margin-top: 5px; font-size: 0.8rem; }
    .action-link { color: #00ffcc !important; text-decoration: none !important; border: 1px solid #00ffcc; padding: 2px 8px; border-radius: 5px; }
    .action-link:hover { background: rgba(0, 255, 204, 0.1); }
    </style>
    """, unsafe_allow_html=True)

# 2. ENGINES & ASSETS
if "GROQ_API_KEY" not in st.secrets:
    st.error("🔑 GROQ_API_KEY missing!")
    st.stop()

@st.cache_resource
def load_assets():
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return client, embedder

client, embedder = load_assets()

# 3. GLOBAL STATE: MULTI-CHAT HISTORY
if "chats" not in st.session_state:
    # Key: chat_id, Value: {title, messages, db, chunks}
    st.session_state.chats = {}

if "active_id" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chats[first_id] = {"title": "Main Brain", "messages": [], "db": None, "chunks": []}
    st.session_state.active_id = first_id

# --- SIDEBAR: NAVIGATION & INDEXING ---
with st.sidebar:
    st.title("🧬 Neural Hub")
    
    if st.button("➕ Start New Session", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Session", "messages": [], "db": None, "chunks": []}
        st.session_state.active_id = new_id
        st.rerun()

    st.divider()
    st.subheader("📚 Chat History")
    for chat_id, chat_data in st.session_state.chats.items():
        # Display session buttons
        label = chat_data["title"] if chat_data["messages"] else "Empty Session"
        if st.button(f"🗨️ {label[:20]}...", key=chat_id, use_container_width=True):
            st.session_state.active_id = chat_id
            st.rerun()

    st.divider()
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
            st.success("Indexed!")

# --- MAIN INTERFACE ---
aid = st.session_state.active_id
active_chat = st.session_state.chats[aid]

st.title("Omni-NCERT Neural Engine")

tab_chat, tab_lab = st.tabs(["💬 AI Tutor Console", "📊 Visual Lab"])

with tab_chat:
    # 1. Display historical messages with actions
    for idx, m in enumerate(active_chat["messages"]):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            
            if m["role"] == "assistant":
                # Create encoded WhatsApp link
                wa_msg = urllib.parse.quote(f"Check out this NCERT Solution: \n\n{m['content'][:500]}...")
                wa_url = f"https://wa.me/?text={wa_msg}"
                
                # Action Buttons Row
                cols = st.columns([0.2, 0.2, 0.2, 0.4])
                with cols[0]: st.button("👍", key=f"like_{idx}_{aid}")
                with cols[1]: st.button("👎", key=f"dis_{idx}_{aid}")
                with cols[2]: st.markdown(f'<a href="{wa_url}" target="_blank" class="action-link">📲 Share</a>', unsafe_allow_html=True)
                with cols[3]: st.code(m["content"][:20] + "...", language="text") # Click code to copy

    # 2. Chat Input Logic
    if prompt := st.chat_input("Ask a question..."):
        # Set chat title based on first question
        if not active_chat["messages"]:
            active_chat["title"] = prompt[:25]

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
            st.rerun() # Refresh to show assistant message with action buttons
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