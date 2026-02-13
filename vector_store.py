import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 📂 Root folder (can be set via environment variable)
PDF_FOLDER = Path(os.getenv("LEGAL_PDF_FOLDER", "legal_docs"))
FAISS_INDEX_PATH = "faiss_index"

# 🔥 Ollama embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# -------------------------
# Empty retriever (fallback)
# -------------------------
class EmptyRetriever:
    def invoke(self, _query):
        return []


# -------------------------
# Load documents (RECURSIVE)
# -------------------------
def load_documents():
    if not PDF_FOLDER.exists():
        print(f"⚠️ PDF folder not found: {PDF_FOLDER.resolve()}")
        return []

    # 🔥 Recursively find all PDFs inside subfolders
    pdf_files = list(PDF_FOLDER.rglob("*.pdf"))

    if not pdf_files:
        print(f"⚠️ No PDF files found in: {PDF_FOLDER.resolve()}")
        return []

    print(f"📂 Found {len(pdf_files)} PDF files (including subfolders)")

    def load_pdf(file_path):
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = file_path.name
            page.metadata["page"] = page.metadata.get("page", 0)
            page.metadata["folder"] = file_path.parent.name
            page.metadata["full_path"] = str(file_path)

        return pages

    # Load PDFs in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_pdf, pdf_files))

    documents = [doc for sublist in results for doc in sublist]

    print(f"📄 Total pages loaded: {len(documents)}")

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    split_docs = splitter.split_documents(documents)

    print(f"✂️ Total chunks created: {len(split_docs)}")

    return split_docs


# -------------------------
# Build Hybrid Retriever
# -------------------------
def _build_retriever(loaded_documents):
    if not loaded_documents:
        return EmptyRetriever(), []

    if os.path.exists(FAISS_INDEX_PATH):
        print("📂 Loading existing FAISS index...")
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        documents = list(vector_store.docstore._dict.values())
    else:
        print("⚡ Building FAISS index...")
        vector_store = FAISS.from_documents(loaded_documents, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        documents = loaded_documents
        print("✅ FAISS index built successfully!")

    # Vector Retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    # BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 6

    # Hybrid Retriever
    hybrid = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )

    return hybrid, documents


# 🔥 Initialize retriever at startup
retriever, documents = _build_retriever(load_documents())


# -------------------------
# Filter by File Name
# -------------------------
def get_filtered_retriever(filename):
    filtered_docs = [
        doc for doc in documents
        if doc.metadata.get("source") == filename
    ]

    if not filtered_docs:
        return EmptyRetriever()

    filtered_vector = FAISS.from_documents(filtered_docs, embeddings)

    vector_ret = filtered_vector.as_retriever(search_kwargs={"k": 4})
    bm25_ret = BM25Retriever.from_documents(filtered_docs)
    bm25_ret.k = 4

    return EnsembleRetriever(
        retrievers=[bm25_ret, vector_ret],
        weights=[0.5, 0.5],
    )


# -------------------------
# Filter by Folder Name
# -------------------------
def get_filtered_retriever_by_folder(folder_name):
    filtered_docs = [
        doc for doc in documents
        if doc.metadata.get("folder") == folder_name
    ]

    if not filtered_docs:
        return EmptyRetriever()

    filtered_vector = FAISS.from_documents(filtered_docs, embeddings)

    vector_ret = filtered_vector.as_retriever(search_kwargs={"k": 4})
    bm25_ret = BM25Retriever.from_documents(filtered_docs)
    bm25_ret.k = 4

    return EnsembleRetriever(
        retrievers=[bm25_ret, vector_ret],
        weights=[0.5, 0.5],
    )
