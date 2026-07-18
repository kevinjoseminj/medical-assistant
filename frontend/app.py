import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

DISCLAIMER = (
    "This assistant provides general health information from public sources "
    "(MedlinePlus, WHO) and is not a substitute for professional medical advice, "
    "diagnosis, or treatment. It does not diagnose conditions or give "
    "personalized medical advice. Always consult a qualified healthcare "
    "provider about your specific situation."
)

st.set_page_config(page_title="Medical Info Assistant", page_icon="\U0001fa7a")
st.title("Medical Info Assistant")
st.warning(DISCLAIMER)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

with st.sidebar:
    st.subheader("Upload a PDF")
    uploaded_file = st.file_uploader(
        "Ask questions about an additional document", type=["pdf"]
    )
    if uploaded_file is not None and uploaded_file.name != st.session_state.uploaded_filename:
        with st.spinner("Processing PDF..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                st.session_state.session_id = data["session_id"]
                st.session_state.uploaded_filename = uploaded_file.name
                st.success(f"Loaded {data['chunk_count']} chunks from {uploaded_file.name}")
            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")
    st.caption(
        "Uploaded documents are not stored — they're kept in memory only for "
        "this session and used alongside the built-in health reference "
        "sources. Answers about them are still informational only, not "
        "diagnostic."
    )


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for source in sources:
            if source.get("url"):
                st.markdown(f"- [{source['title']}]({source['url']})")
            else:
                st.markdown(f"- {source['title']}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

question = st.chat_input("Ask a general health question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={
                        "question": question,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as exc:
                st.error(f"Couldn't reach the backend: {exc}")
                st.stop()

        st.markdown(data["answer"])
        render_sources(data["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "content": data["answer"], "sources": data["sources"]}
    )
