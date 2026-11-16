"""
app.py
------
Streamlit Web Application for the RAG-Based Intelligent Academic Advisor System.

Key Features:
- Loads local FAISS vectorstore ('vectorstore_db') with HuggingFaceEmbeddings.
- Provides interactive student advising chat interface with persistent session memory.
- Supports flexible LLM providers (OpenAI, Cloud API, or Retrieval-Only Mode).
- Enforces strict grounded generation to eliminate hallucinations regarding academic policies.
- Displays retrieved source catalog citations with similarity scores for full transparency.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# LangChain Imports
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Constants
DEFAULT_VECTORSTORE_DIR = "vectorstore_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Page configuration
st.set_page_config(
    page_title="Intelligent Academic Advisor | Student Support RAG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished university advisory appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.88rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    .stChatMessage {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading semantic embedding model (HuggingFace all-MiniLM-L6-v2)...")
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Caches and returns the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


@st.cache_resource(show_spinner="Loading FAISS vector database...")
def load_faiss_vectorstore(vectorstore_dir: str) -> Optional[FAISS]:
    """Loads existing FAISS vector store from local directory."""
    path = Path(vectorstore_dir)
    if not path.exists() or not (path / "index.faiss").exists():
        return None
    
    embeddings = get_embedding_model()
    try:
        vectorstore = FAISS.load_local(
            folder_path=str(path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True  # Trusted local vector store
        )
        return vectorstore
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None


def get_llm(provider: str, api_key: str, model_name: str, temperature: float = 0.1):
    """Initializes LLM instance based on selected provider."""
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            api_key=api_key,
            temperature=temperature
        )
    return None


def format_docs(docs: List[Document]) -> str:
    """Formats retrieved Document objects into a single context string."""
    formatted = []
    for idx, doc in enumerate(docs, 1):
        source_name = doc.metadata.get("source", "Academic Catalog")
        formatted.append(f"[Source {idx} - {source_name}]:\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)


def build_rag_prompt() -> ChatPromptTemplate:
    """Constructs a strict grounding prompt for academic policy advising."""
    system_prompt = (
        "You are an empathetic, authoritative, and precise University Academic Advisor AI assisting university students.\n"
        "Your task is to answer student questions about degree requirements, course prerequisites, registration rules, "
        "and academic standing STRICTLY based on the provided official university catalog excerpts.\n\n"
        "GUIDELINES:\n"
        "1. Strictly base your answers ONLY on the provided Context excerpts below.\n"
        "2. If the context does not contain enough information to answer definitively, state:\n"
        "   'I cannot find specific information regarding this query in the current university academic catalog. "
        "Please contact your assigned academic advisor or the Registrar's Office for authoritative guidance.'\n"
        "3. Explicitly cite specific courses (e.g., 'CS 201'), minimum grades (e.g., 'C or higher'), "
        "GPA cutoffs, and policy section titles mentioned in the context.\n"
        "4. Be structured, encouraging, and clear using bullet points or numbered lists where appropriate.\n"
        "5. Do NOT hallucinate prerequisites, course numbers, or university rules.\n\n"
        "Context:\n{context}\n"
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])


def execute_rag_query(
    question: str, 
    vectorstore: FAISS, 
    top_k: int, 
    llm=None
) -> Tuple[str, List[Tuple[Document, float]]]:
    """
    Performs similarity search against FAISS, then generates a grounded LLM response
    or returns formatted similarity findings if LLM is disabled.
    """
    # Retrieve documents with distance scores
    retrieved_with_scores = vectorstore.similarity_search_with_score(question, k=top_k)
    retrieved_docs = [doc for doc, score in retrieved_with_scores]

    if not retrieved_docs:
        return (
            "No relevant sections in the academic catalog could be matched with your inquiry.",
            []
        )

    # If no LLM is configured, run in Retrieval/Similarity Search Inspection mode
    if llm is None:
        context_preview = "\n\n---\n\n".join([
            f"**Section Excerpt #{i+1}** (Relevance Distance: `{score:.4f}`):\n\n{doc.page_content}"
            for i, (doc, score) in enumerate(retrieved_with_scores)
        ])
        direct_answer = (
            "ℹ️ **Running in Retrieval-Only Mode (No LLM API Key Connected)**\n\n"
            "Here are the most relevant catalog excerpts retrieved for your query:\n\n"
            f"{context_preview}\n\n"
            "💡 *To generate synthesized conversational answers, enter an API key in the sidebar.*"
        )
        return direct_answer, retrieved_with_scores

    # Execute Modern LangChain LCEL RAG Chain
    prompt = build_rag_prompt()
    rag_chain = (
        {
            "context": lambda x: format_docs(retrieved_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    response_text = rag_chain.invoke(question)
    return response_text, retrieved_with_scores


# -------------------------------------------------------------
# Main Streamlit Application
# -------------------------------------------------------------
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
        st.title("Advisor Settings")

        # Vectorstore status check
        vectorstore = load_faiss_vectorstore(DEFAULT_VECTORSTORE_DIR)
        if vectorstore is not None:
            st.success("✅ FAISS Vector Index Loaded")
            st.caption(f"📁 Directory: `{DEFAULT_VECTORSTORE_DIR}`")
            st.caption(f"🧠 Model: `{EMBEDDING_MODEL_NAME}`")
        else:
            st.error("❌ Vector Store Not Found!")
            st.warning("Please run the ingestion script first:\n```bash\npython ingest_and_index.py\n```")

        st.divider()

        # LLM Provider Configuration
        st.subheader("🤖 LLM Reasoning Engine")
        provider = st.selectbox(
            "Select Provider",
            ["OpenAI", "Retrieval-Only (No API Key)"],
            index=0
        )

        api_key = ""
        model_name = ""

        if provider == "OpenAI":
            default_key = os.getenv("OPENAI_API_KEY", "")
            api_key = st.text_input("OpenAI API Key", value=default_key, type="password")
            model_name = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        else:
            st.info("Direct similarity search mode will be used. No external API calls are made.")

        st.divider()

        # Retrieval Settings
        st.subheader("🔍 Retrieval Parameters")
        top_k = st.slider("Catalog Chunks to Retrieve (top-k)", min_value=1, max_value=8, value=3)

        st.divider()

        # Sample Quick Questions
        st.subheader("💡 Sample Student Queries")
        preset_questions = [
            "What are the prerequisites for CS 301 (Algorithms)?",
            "What are the prerequisites for Operating Systems (CS 305)?",
            "What is the minimum GPA required to graduate?",
            "What happens if my cumulative GPA falls below 2.00?",
            "What is the deadline for dropping a course without penalty?",
            "How does the course retake and grade replacement policy work?",
            "How many credits do I need to graduate with a BS in CS?"
        ]

        for q in preset_questions:
            if st.button(q, use_container_width=True):
                st.session_state["user_input_preset"] = q

        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Main UI Header
    st.markdown('<div class="main-header">🎓 Intelligent Academic Advisor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'AI-Powered Student Advising System strictly grounded in the official university academic catalog.'
        '</div>', 
        unsafe_allow_html=True
    )

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your University Academic Advisor AI. "
                    "I can answer questions regarding course prerequisites, academic probation, "
                    "graduation requirements, and add/drop deadlines based on the official academic catalog. "
                    "How can I help you plan your academic journey today?"
                ),
                "sources": []
            }
        ]

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Retrieved Academic Catalog Sources", expanded=False):
                    for idx, (doc, score) in enumerate(msg["sources"], 1):
                        source_file = doc.metadata.get("source", "Catalog")
                        st.markdown(
                            f"<div class='source-box'>"
                            f"<b>Citation #{idx}</b> (Source: <code>{source_file}</code> | Distance: <code>{score:.4f}</code>)<br>"
                            f"{doc.page_content}"
                            f"</div>",
                            unsafe_allow_html=True
                        )

    # Handle Input from either preset button or chat input
    user_prompt = None
    if "user_input_preset" in st.session_state and st.session_state["user_input_preset"]:
        user_prompt = st.session_state.pop("user_input_preset")
    else:
        chat_val = st.chat_input("Ask about prerequisites, degree requirements, or academic policies...")
        if chat_val:
            user_prompt = chat_val

    # Process query
    if user_prompt:
        if vectorstore is None:
            st.error("Vector database is not loaded. Please run `python ingest_and_index.py` first.")
            return

        # Append and display student question
        st.session_state.messages.append({"role": "user", "content": user_prompt, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Reviewing academic catalog policies and prerequisites..."):
                try:
                    # Initialize LLM if requested
                    llm = None
                    if provider != "Retrieval-Only (No API Key)":
                        if not api_key:
                            st.warning(f"⚠️ Please enter an API key for {provider} in the sidebar, or switch to 'Retrieval-Only' mode.")
                            return
                        llm = get_llm(provider, api_key, model_name)

                    # Query RAG
                    answer, sources = execute_rag_query(
                        question=user_prompt,
                        vectorstore=vectorstore,
                        top_k=top_k,
                        llm=llm
                    )

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 Retrieved Academic Catalog Sources", expanded=False):
                            for idx, (doc, score) in enumerate(sources, 1):
                                source_file = doc.metadata.get("source", "Catalog")
                                st.markdown(
                                    f"<div class='source-box'>"
                                    f"<b>Citation #{idx}</b> (Source: <code>{source_file}</code> | Distance: <code>{score:.4f}</code>)<br>"
                                    f"{doc.page_content}"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )

                    # Save to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Error processing inquiry: {e}")


if __name__ == "__main__":
    main()
