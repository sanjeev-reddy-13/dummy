# main.py

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_store import retriever, get_filtered_retriever

model = OllamaLLM(model="llama3.2")

template = """
You are a professional legal assistant.

Answer ONLY using the provided context.
If the answer is not found, say:
"I could not find this information in the provided documents."

Context:
{context}

Question:
{question}

Provide a clear legal explanation.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n-----------------------------------")
    question = input("Ask a legal question (q to quit): ")

    if question.lower() == "q":
        break

    # 🔥 OPTIONAL: Filter by specific PDF
    # filename = "contract1.pdf"
    # filtered_retriever = get_filtered_retriever(filename)
    # docs = filtered_retriever.invoke(question)

    # Default: search all documents
    docs = retriever.invoke(question)

    # Combine context
    context = "\n\n".join([doc.page_content for doc in docs])

    result = chain.invoke({
        "context": context,
        "question": question
    })

    print("\n📌 Answer:\n")
    print(result)

    # 🔥 Print Sources
    print("\n📚 Sources:")
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        print(f"- Page {page} of {source}")
