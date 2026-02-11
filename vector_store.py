# vector_store.py

import os
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever


PDF_FOLDER = r"D:\Projects\LegalAi\legal_docs"
FAISS_INDEX_PATH = "faiss_index"

# 🔥 Faster embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# -----------------------------------------
# LOAD & SPLIT DOCUMENTS
# -----------------------------------------

def load_documents():

    def load_pdf(file):
        loader = PyMuPDFLoader(os.path.join(PDF_FOLDER, file))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = file
            page.metadata["page"] = page.metadata.get("page", 0)

        return pages

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_pdf, pdf_files))

    documents = [doc for sublist in results for doc in sublist]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    return text_splitter.split_documents(documents)


# -----------------------------------------
# BUILD OR LOAD FAISS
# -----------------------------------------

if os.path.exists(FAISS_INDEX_PATH):

    print("📂 Loading existing FAISS index...")
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    documents = vector_store.docstore._dict.values()

else:

    print("⚡ Building FAISS index...")
    split_docs = load_documents()

    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)

    documents = split_docs

    print("✅ FAISS index built successfully!")


# -----------------------------------------
# CREATE RETRIEVERS
# -----------------------------------------

# 🚀 Vector Retriever
vector_retriever = vector_store.as_retriever(
    search_kwargs={"k": 6}
)

# 🧠 BM25 Retriever (Keyword Search)
bm25_retriever = BM25Retriever.from_documents(list(documents))
bm25_retriever.k = 6

# 🔥 Hybrid Retriever (Weighted)
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # You can tune this
)

# 🔎 Optional file filter
def get_filtered_retriever(filename):
    filtered_docs = [
        doc for doc in documents
        if doc.metadata.get("source") == filename
    ]

    filtered_vector = FAISS.from_documents(filtered_docs, embeddings)

    vector_ret = filtered_vector.as_retriever(search_kwargs={"k": 4})
    bm25_ret = BM25Retriever.from_documents(filtered_docs)
    bm25_ret.k = 4

    return EnsembleRetriever(
        retrievers=[bm25_ret, vector_ret],
        weights=[0.5, 0.5]
    )
