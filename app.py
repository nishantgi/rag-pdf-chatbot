import streamlit as st
import os
from dotenv import load_dotenv
from src.rag_pipeline1 import rag_query
from src.vector_store import VectorStore
from src.demo_rag_index import build_full_index  # For re-indexing

load_dotenv()

st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")

# Sidebar
st.sidebar.title("⚙️ RAG Chatbot Controls")
if st.sidebar.button("🔄 Rebuild Index"):
    with st.spinner("Rebuilding index..."):
        build_full_index()
    st.sidebar.success("✅ Index rebuilt!")

# Load index
@st.cache_resource
def load_rag_store():
    return VectorStore.load_index()

store = load_rag_store()
if not store:
    st.error("❌ No index found! Click 'Rebuild Index' in sidebar.")
    st.stop()
st.rerun()
# Chat interface
st.title("📚 RAG PDF Chatbot")
st.markdown("**Ask questions about your DSA PDF** (104 chunks indexed)")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about QuickSort, Linked Lists, Trees..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # RAG + LLM
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag_query(prompt, store)
            response = result["answer"]
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("*Built with FAISS + SentenceTransformers + Groq LLM*")
