from typing import Callable, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from backend.schemas import SourceRef


class AnswerOutput(BaseModel):
    answer: str = Field(
        description=(
            "The answer to the user's question, or a brief refusal/redirect "
            "message if the question asks for a diagnosis or personalized "
            "medical advice."
        )
    )
    grounded_in_context: bool = Field(
        description=(
            "True only if the answer draws on facts from the provided context "
            "documents. False for refusals, redirects, or 'not enough "
            "information' responses that do not cite the context."
        )
    )

DISCLAIMER = (
    "This tool provides general health information from public sources "
    "(MedlinePlus, WHO) and is not a substitute for professional medical "
    "advice, diagnosis, or treatment. Always consult a qualified healthcare "
    "provider about your specific situation."
)

SYSTEM_PROMPT = """You are a health-information assistant. Answer using ONLY \
the context below. It may include public health reference sources \
(MedlinePlus, WHO) and/or a document the user uploaded.

Rules:
- Only answer general informational questions ("what is X", "what are common \
symptoms/causes of Y"). Do not use knowledge beyond the given context.
- If the context doesn't contain enough information to answer, say so plainly \
instead of guessing.
- If the user describes their own symptoms, asks for a diagnosis, or asks what \
they personally should do, do not diagnose or give personalized medical advice. \
Explain that you can't do that and recommend they consult a healthcare provider. \
You may still share general context-backed information on the topic if relevant.
- If the user asks you to judge whether their own personal reading, lab \
result, vital sign, or measurement (including values found in an uploaded \
document) is normal, abnormal, good, or concerning, do NOT render that \
judgment in any form. You may state the general reference range for that \
measurement from the context (e.g. "normal fasting blood sugar is typically \
under 100 mg/dL"), but do NOT restate the user's own value alongside a \
verdict word or phrase about it — no "which is normal", "falls within the \
normal range", "is elevated", "is within/outside range", or similar, even \
softened. State the range on its own and stop there; let the user draw any \
comparison themselves. Make clear that assessing a specific personal value \
requires a healthcare provider, not you.
- Never claim to be a doctor or medical professional.
- Be concise.

Context:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def format_sources(docs: list[Document]) -> list[SourceRef]:
    seen: set[str] = set()
    sources = []
    for doc in docs:
        url = doc.metadata.get("url", "")
        if url in seen:
            continue
        seen.add(url)
        sources.append(SourceRef(title=doc.metadata.get("title", ""), url=url))
    return sources


def build_chain(
    base_retriever: Runnable,
    get_session_retriever: Callable[[str], Optional[Runnable]],
) -> Runnable:
    """Build a chain: {"question": str, "session_id": str | None} -> {"answer": str, "docs": list[Document]}.

    Retrieval always includes the base corpus; when ``session_id`` maps to a
    live upload session (via ``get_session_retriever``), that session's docs
    are appended so an uploaded PDF is queryable alongside the base corpus.

    ``docs`` in the output is only populated when the model reports it
    actually grounded the answer in the retrieved context — refusals/
    redirects (e.g. diagnosis requests) return an empty source list instead
    of citing whatever the retriever happened to return for an off-corpus
    query.
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_model = model.with_structured_output(AnswerOutput)
    answer_chain = PROMPT | structured_model

    def retrieve(inputs: dict) -> dict:
        question = inputs["question"]
        docs = list(base_retriever.invoke(question))

        session_id = inputs.get("session_id")
        session_retriever = get_session_retriever(session_id) if session_id else None
        if session_retriever is not None:
            docs = docs + list(session_retriever.invoke(question))

        return {"question": question, "docs": docs}

    def generate(inputs: dict) -> dict:
        docs = inputs["docs"]
        result: AnswerOutput = answer_chain.invoke(
            {"context": format_docs(docs), "question": inputs["question"]}
        )
        return {"answer": result.answer, "docs": docs if result.grounded_in_context else []}

    return RunnableLambda(retrieve) | RunnableLambda(generate)
