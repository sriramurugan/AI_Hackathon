import streamlit as st
import numpy as np
import faiss
from groq import Groq
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

st.set_page_config(page_title="NCERT AI Tutor")
st.title("📚 NCERT AI Tutor")

@st.cache_resource
def load_models():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_models()

if "chunks" not in st.session_state: st.session_state.chunks = []
if "index" not in st.session_state: st.session_state.index = None

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

pdf = st.file_uploader("Upload NCERT PDF", type="pdf")

if pdf:
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t: text += t
    
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    st.session_state.chunks = chunks
    embeddings = embed_model.encode(chunks)
    
    # FIX 1: Ensure float32 for FAISS compatibility
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    
    st.session_state.index = index
    st.success("PDF indexed successfully.")

question = st.chat_input("Ask NCERT question")

if question:
    st.write(f"### Question: {question}")
    context = ""

    # FIX 2: Check if index exists before searching to avoid crash
    if st.session_state.index is not None:
        q_embed = embed_model.encode([question])
        D, I = st.session_state.index.search(np.array(q_embed).astype('float32'), 3)
        retrieved = [st.session_state.chunks[i] for i in I[0]]
        context = "\n".join(retrieved)
    else:
        context = "No PDF uploaded. Answer based on general NCERT knowledge."

    # FIX 3: Stable Model Name & System Prompt
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nExplain step-by-step with formulas."
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        st.write("### Answer")
        st.write(answer)
    except Exception as e:
        st.error(f"Groq API Error: {e}")

    # --- Graph Logic ---
    q = question.lower()
    if "plot" in q or "graph" in q:
        try:
            x = np.linspace(-10, 10, 100)
            if "x^2" in q or "x²" in q: y = x**2
            elif "2x+3" in q: y = 2*x + 3
            else: y = x
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_title("Generated Graph")
            ax.grid(True)
            st.pyplot(fig)
        except:
            st.warning("Graph could not be generated.")