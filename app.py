import streamlit as st
import numpy as np
import faiss
from groq import Groq
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

st.set_page_config(page_title="NCERT AI Tutor")

st.title("📚 NCERT AI Tutor")

# ---------- Load models ----------
@st.cache_resource
def load_models():
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    return embed

embed_model = load_models()

# ---------- Initialize session ----------
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

# ---------- Groq client ----------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# ---------- Upload PDF ----------
pdf = st.file_uploader("Upload NCERT PDF", type="pdf")

if pdf:

    reader = PdfReader(pdf)

    text = ""

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t

    # chunk text
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    st.session_state.chunks = chunks

    # create embeddings
    embeddings = embed_model.encode(chunks)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    st.session_state.index = index

    st.success("PDF indexed successfully.")


# ---------- Chat input ----------
question = st.chat_input("Ask NCERT question")

if question:

    st.write("### Question")
    st.write(question)

    context = ""

    if st.session_state.index is not None:

        q_embed = embed_model.encode([question])

        D, I = st.session_state.index.search(np.array(q_embed), 3)

        retrieved = [st.session_state.chunks[i] for i in I[0]]

        context = "\n".join(retrieved)

    prompt = f"""
You are an expert NCERT tutor for Physics, Chemistry and Mathematics.

Context from NCERT:
{context}

Question:
{question}

Explain step-by-step with formulas when needed.
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    st.write("### Answer")
    st.write(answer)


    # ---------- Graph Generation ----------
    q = question.lower()

    if "plot" in q or "graph" in q:

        try:

            x = np.linspace(-10, 10, 100)

            if "x^2" in q or "x²" in q:
                y = x**2

            elif "2x+3" in q:
                y = 2*x + 3

            else:
                y = x

            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_title("Generated Graph")
            ax.grid(True)

            st.pyplot(fig)

        except:
            st.warning("Graph could not be generated.")