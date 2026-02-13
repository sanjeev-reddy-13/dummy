import json
from sentence_transformers import CrossEncoder

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_store import retrievers


# ======================================================
# MODEL
# ======================================================
model = OllamaLLM(model="llama3.2")

AVAILABLE_DOMAINS = list(retrievers.keys())
domains_text = "\n".join(AVAILABLE_DOMAINS)

# ======================================================
# RERANKER
# ======================================================
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ======================================================
# AUTONOMOUS QUERY PLANNER
# ======================================================
planner_template = """
You are an expert legal AI planner.

Available domains:
{domains}

Return JSON ONLY in this format:

{{
  "domains": ["Domain Name"],
  "queries": ["query1","query2","query3"]
}}

Question:
{question}
"""

planner_chain = (
    ChatPromptTemplate.from_template(planner_template)
    | model
)


# ======================================================
# ANSWER GENERATION
# ======================================================
answer_template = """
You are a professional legal assistant.

Answer ONLY from provided context.
Do not hallucinate.

Context:
{context}

Question:
{question}

Give clear legal explanation.
Mention acts/sections if present.
"""

answer_chain = (
    ChatPromptTemplate.from_template(answer_template)
    | model
)


# ======================================================
# MAIN LOOP
# ======================================================
while True:

    print("\n-----------------------------------")
    question = input("Ask legal question (q to quit): ")

    if question.lower() == "q":
        break

    # ---------- Planner ----------
    try:
        raw_plan = planner_chain.invoke({
            "question": question,
            "domains": domains_text
        })

        plan = json.loads(raw_plan)

    except Exception:
        print("⚠️ Planner failed → fallback")
        plan = {
            "domains": AVAILABLE_DOMAINS,
            "queries": [question]
        }

    domains = plan.get("domains", AVAILABLE_DOMAINS)
    queries = plan.get("queries", [question])

    print("\n🧠 Domains:")
    for d in domains:
        print("-", d)

    print("\n🔎 Queries:")
    for q in queries:
        print("-", q)

    # ---------- Retrieval ----------
    all_docs = []

    for domain in domains:
        retriever = retrievers.get(domain)

        if retriever:
            for q in queries:
                docs = retriever.invoke(q)
                all_docs.extend(docs)

    # Remove duplicates
    unique_docs = list(
        {d.page_content: d for d in all_docs}.values()
    )

    if not unique_docs:
        print("⚠️ No documents found")
        continue

    # ---------- RERANK ----------
    print("\n⚡ Reranking...")

    pairs = [(question, d.page_content) for d in unique_docs]
    scores = reranker.predict(pairs)

    scored = list(zip(unique_docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    top_docs = [d for d, _ in scored[:5]]

    context = "\n\n".join(d.page_content for d in top_docs)

    # ---------- Answer ----------
    result = answer_chain.invoke({
        "context": context,
        "question": question
    })

    print("\n📌 FINAL ANSWER:\n")
    print(result)

    print("\n📚 Sources:")
    for d in top_docs:
        print(
            f"- Page {d.metadata.get('page')} | "
            f"{d.metadata.get('source')} | "
            f"{d.metadata.get('domain')}"
        )
