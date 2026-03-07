import streamlit as st
from groq import Groq
import matplotlib.pyplot as plt
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="NCERT Hybrid AI Tutor", page_icon="📚")
st.title("📚 NCERT Hybrid AI Learning Tutor")

# 2. Secure API Key Loading
# This looks in Streamlit Secrets (Cloud) or environment variables
if "GROQ_API_KEY" not in st.secrets:
    st.error("Missing GROQ_API_KEY in Streamlit Secrets!")
    st.stop()

# 3. Initialize the Groq Client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# 4. Initialize CHAT MEMORY (The Notebook)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful NCERT Science and Math tutor. Use clear examples."}
    ]

# 5. Display the Chat History (so words don't disappear)
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 6. The "Chat Input" and AI Logic
if prompt := st.chat_input("Ask me a science or math question..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save user message to memory
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call Groq with the ENTIRE history (This gives it memory!)
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=st.session_state.messages, # Sending the whole notebook!
            temperature=0.7,
        )
        
        response = completion.choices[0].message.content
        
        # Show assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Save assistant response to memory
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Groq API Error: {e}")

# 7. Sidebar for Visualization (Fixing Matplotlib)
with st.sidebar:
    st.header("Visual Tools")
    if st.button("Generate Science Graph"):
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, color='green')
        ax.set_title("Visualizing a Sine Wave (Physics)")
        st.pyplot(fig) # Correct way to show matplotlib in Streamlit