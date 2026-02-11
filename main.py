from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_store import retriever, get_filtered_retriever


# 🔥 Local LLM
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

Provide a clear legal explanation in simple language.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


while True:
    print("\n-----------------------------------")
    question = input("Ask a legal question (q to quit): ")

    if question.lower() == "q":
        break

    docs = retriever.invoke(question)

    if not docs:
        print("⚠️ No relevant documents found.")
        continue

    context = "\n\n".join([doc.page_content for doc in docs])

    result = chain.invoke({
        "context": context,
        "question": question
    })

    print("\n📌 Answer:\n")
    print(result)

    print("\n📚 Sources:")
    for doc in docs:
        print(f"- Page {doc.metadata.get('page')} of {doc.metadata.get('source')}")
