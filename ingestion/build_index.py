"""Fetch curated public health sources, split, embed, and persist a FAISS index.

Standalone and re-runnable:

    uv run python -m ingestion.build_index

Not invoked at request time — the backend loads the resulting directory
(``VECTORSTORE_DIR``, default ``vectorstore/``) once at startup.
"""

import html
import os
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.sources import EXTRA_SOURCES, MEDLINEPLUS_TOPICS

MEDLINEPLUS_ENDPOINT = "https://wsearch.nlm.nih.gov/ws/query"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def fetch_medlineplus_topic(term: str) -> Document | None:
    """Query the MedlinePlus web service and return the top-ranked topic as a Document."""
    params = {"db": "healthTopics", "term": term, "rettype": "topic", "retmax": 1}
    response = requests.get(MEDLINEPLUS_ENDPOINT, params=params, timeout=15)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    topic_el = root.find(".//health-topic")
    if topic_el is None:
        return None

    title = topic_el.get("title", term)
    url = topic_el.get("url", "")
    summary_el = topic_el.find("full-summary")
    if summary_el is None or not summary_el.text:
        return None

    # full-summary content is HTML, HTML-entity-escaped inside the XML text node.
    raw_html = html.unescape(summary_el.text)
    text = BeautifulSoup(raw_html, "lxml").get_text(separator="\n", strip=True)
    if not text:
        return None

    return Document(
        page_content=f"{title}\n\n{text}",
        metadata={"source": "MedlinePlus", "title": title, "url": url},
    )


def fetch_html_page(url: str, label: str) -> Document | None:
    """Fetch an HTML fact sheet page and extract its main text content."""
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "lxml")

    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    text = main.get_text(separator="\n", strip=True) if main else ""
    if not text:
        return None

    title_el = soup.find("title")
    title = title_el.get_text(strip=True) if title_el else label

    return Document(
        page_content=f"{title}\n\n{text}",
        metadata={"source": label.split(":")[0], "title": title, "url": url},
    )


def build_documents() -> list[Document]:
    documents: list[Document] = []
    failures: list[str] = []

    for term in MEDLINEPLUS_TOPICS:
        try:
            doc = fetch_medlineplus_topic(term)
            if doc is None:
                failures.append(f"MedlinePlus:{term} (no full-summary found)")
                continue
            documents.append(doc)
        except requests.RequestException as exc:
            failures.append(f"MedlinePlus:{term} ({exc})")

    for url, label in EXTRA_SOURCES:
        try:
            doc = fetch_html_page(url, label)
            if doc is None:
                failures.append(f"{label} (no content extracted)")
                continue
            documents.append(doc)
        except requests.RequestException as exc:
            failures.append(f"{label} ({exc})")

    print(f"Fetched {len(documents)} documents ({len(failures)} failures).")
    for failure in failures:
        print(f"  FAILED: {failure}")

    return documents


def main() -> None:
    load_dotenv()
    vectorstore_dir = os.environ.get("VECTORSTORE_DIR", "vectorstore")

    documents = build_documents()
    if not documents:
        raise SystemExit("No documents fetched — aborting index build.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(vectorstore_dir)
    print(f"Saved FAISS index to '{vectorstore_dir}/'.")


if __name__ == "__main__":
    main()
