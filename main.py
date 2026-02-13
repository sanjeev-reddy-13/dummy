from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_store import retriever

# 🔥 Local LLM
model = OllamaLLM(model="llama3.2")

# ==========================================================
# 1️⃣ QUERY EXPANSION / UNDERSTANDING LAYER
# ==========================================================

rewrite_template = """
You are a legal query expansion system.

Generate 3 different search queries for the following question:
1. A simple clearer version
2. A formal legal terminology version
3. A version including related legal keywords

Return ONLY the 3 queries separated by newline.
Do not add numbering.

User Question:
{question}
"""

rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
rewrite_chain = rewrite_prompt | model


# ==========================================================
# 2️⃣ ANSWER GENERATION PROMPT
# ==========================================================

answer_template = """
You are a professional legal assistant.

Answer ONLY using the provided context.
If the answer is not found, say:
"I could not find this information in the provided documents."

Context:
{context}

Question:
{question}

Provide a clear legal explanation in simple language.
Mention relevant sections or articles if present.
"""

answer_prompt = ChatPromptTemplate.from_template(answer_template)
answer_chain = answer_prompt | model


# ==========================================================
# 3️⃣ INTERACTIVE LOOP
# ==========================================================

while True:
    print("\n-----------------------------------")
    question = input("Ask a legal question (q to quit): ")

    if question.lower() == "q":
        break

    # ------------------------------------------
    # 🔥 Step 1: Expand the query
    # ------------------------------------------
    expanded = rewrite_chain.invoke({
        "question": question
    })

    print("\n🔎 Expanded Search Queries:\n")
    print(expanded)

    queries = [q.strip() for q in expanded.split("\n") if q.strip()]

    # ------------------------------------------
    # 🔥 Step 2: Retrieve using multiple queries
    # ------------------------------------------
    all_docs = []

    for q in queries:
        docs = retriever.invoke(q)
        all_docs.extend(docs)

    # Remove duplicate chunks
    unique_docs = list({
        doc.page_content: doc for doc in all_docs
    }.values())

    docs = unique_docs[:6]

    if not docs:
        print("⚠️ No relevant documents found.")
        continue

    # ------------------------------------------
    # 🔥 Step 3: Prepare context
    # ------------------------------------------
    context = "\n\n".join([doc.page_content for doc in docs])

    # ------------------------------------------
    # 🔥 Step 4: Generate final answer
    # ------------------------------------------
    result = answer_chain.invoke({
        "context": context,
        "question": question
    })

    print("\n📌 Answer:\n")
    print(result)

    # ------------------------------------------
    # 🔥 Step 5: Show Sources
    # ------------------------------------------
    print("\n📚 Sources:")
    for doc in docs:
        print(
            f"- Page {doc.metadata.get('page')} | "
            f"{doc.metadata.get('source')} | "
            f"Folder: {doc.metadata.get('folder')}"
        )
