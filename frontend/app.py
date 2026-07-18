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


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for source in sources:
            st.markdown(f"- [{source['title']}]({source['url']})")


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
                    f"{BACKEND_URL}/query", json={"question": question}, timeout=30
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
