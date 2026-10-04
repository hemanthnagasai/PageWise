import os

import streamlit as st
from dotenv import load_dotenv

import rag_pipeline as rag

load_dotenv()

st.set_page_config(page_title="PageWise", page_icon="📄", layout="wide")

# Streamlit re-runs this whole file on every click, so anything I want to
# survive between runs has to live in session_state.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_meta" not in st.session_state:
    st.session_state.doc_meta = None
if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None


def render_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.markdown(f"**Page {source['page']}**")
            st.caption(source["snippet"])
            if source["has_visual"]:
                st.caption("This page had a table or chart that the vision model described.")
            if source["image_path"] and os.path.exists(source["image_path"]):
                st.image(source["image_path"], width=280)
            st.divider()


with st.sidebar:
    st.title("📄 PageWise")
    st.caption("Upload a PDF, then ask it questions.")

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY is not set. Copy .env.example to .env, add your Gemini key, then restart.")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_id = f"{uploaded_file.name}:{len(file_bytes)}"

        # Without this check the file gets re-ingested on every rerun, which
        # means on every single chat message.
        if file_id != st.session_state.processed_file_id:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def on_progress(current, total, message):
                progress_bar.progress(min(current / total, 1.0))
                status_text.caption(message)

            with st.spinner("Processing document..."):
                try:
                    result = rag.ingest_pdf(file_bytes, uploaded_file.name, progress_callback=on_progress)
                    st.session_state.vector_store = result.vector_store
                    st.session_state.doc_meta = {
                        "filename": result.filename,
                        "num_pages": result.num_pages,
                        "num_chunks": result.num_chunks,
                    }
                    st.session_state.processed_file_id = file_id
                    st.session_state.messages = []
                    progress_bar.progress(1.0)
                    status_text.caption("Done.")
                except Exception as e:
                    st.error(f"Failed to process document: {e}")

    if st.session_state.doc_meta:
        meta = st.session_state.doc_meta
        st.success(f"**{meta['filename']}**\n\n{meta['num_pages']} pages indexed as {meta['num_chunks']} chunks.")
        st.divider()

        st.session_state.retriever_k = st.slider("Chunks retrieved per question (k)", 1, 10, rag.RETRIEVER_K)

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

st.header("Chat with your document")

if st.session_state.vector_store is None:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["sources"]:
            render_sources(message["sources"])

question = st.chat_input("Ask a question about the document...")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            sources = []
            try:
                k = st.session_state.get("retriever_k", rag.RETRIEVER_K)
                result = rag.ask(st.session_state.vector_store, question, k=k)
                answer = result["answer"]

                for doc in result["source_documents"]:
                    snippet = doc.page_content[:400]
                    if len(doc.page_content) > 400:
                        snippet = snippet + "..."
                    sources.append({
                        "page": doc.metadata.get("page", "?"),
                        "snippet": snippet,
                        "image_path": doc.metadata.get("image_path"),
                        "has_visual": doc.metadata.get("has_visual", False),
                    })
            except Exception as e:
                answer = f"Something went wrong answering that: {e}"

        st.markdown(answer)
        if sources:
            render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
