import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings


# ======================================================
# CONFIG
# ======================================================
BASE_DOC_FOLDER = Path("legal_docs")
INDEX_FOLDER = Path("faiss_indexes")

embeddings = OllamaEmbeddings(model="nomic-embed-text")


class EmptyRetriever:
    def invoke(self, _query):
        return []


# ======================================================
# LOAD DOCUMENTS
# ======================================================
def load_domain_documents(domain_folder):

    pdf_files = list(domain_folder.rglob("*.pdf"))
    if not pdf_files:
        return []

    print(f"\n📂 Domain: {domain_folder.name}")
    print(f"📄 PDFs: {len(pdf_files)}")

    def load_pdf(file):
        loader = PyPDFLoader(str(file))
        pages = loader.load()

        for p in pages:
            p.metadata["source"] = file.name
            p.metadata["domain"] = domain_folder.name
            p.metadata["page"] = p.metadata.get("page", 0)

        return pages

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_pdf, pdf_files))

    docs = [d for sub in results for d in sub]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(docs)

    print(f"✂️ Chunks: {len(chunks)}")
    return chunks


# ======================================================
# BUILD / UPDATE INDEX (INCREMENTAL)
# ======================================================
def build_domain_index(domain_folder):

    docs = load_domain_documents(domain_folder)

    if not docs:
        return EmptyRetriever()

    index_path = INDEX_FOLDER / domain_folder.name
    index_path.mkdir(parents=True, exist_ok=True)

    tracker_file = index_path / "indexed_files.json"

    # --- load indexed files ---
    if tracker_file.exists():
        indexed_files = set(json.loads(tracker_file.read_text()))
    else:
        indexed_files = set()

    current_files = set(d.metadata["source"] for d in docs)
    new_files = current_files - indexed_files

    new_docs = [
        d for d in docs
        if d.metadata["source"] in new_files
    ]

    # --- Load or Create FAISS ---
    if (index_path / "index.faiss").exists():

        print(f"📂 Loading FAISS → {domain_folder.name}")

        vs = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        if new_docs:
            print(f"➕ Adding new docs: {len(new_docs)} chunks")
            vs.add_documents(new_docs)
            vs.save_local(str(index_path))

    else:
        print(f"⚡ Creating FAISS → {domain_folder.name}")
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(str(index_path))
        new_files = current_files

    # --- save tracker ---
    tracker_file.write_text(
        json.dumps(list(indexed_files | new_files))
    )

    print(f"✅ Indexed files: {len(indexed_files | new_files)}")

    # --- Hybrid Retrieval ---
    vector_ret = vs.as_retriever(search_kwargs={"k": 40})

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 40

    hybrid = EnsembleRetriever(
        retrievers=[bm25, vector_ret],
        weights=[0.6, 0.4]
    )

    return hybrid


# ======================================================
# LOAD ALL RETRIEVERS
# ======================================================
def load_all_retrievers():

    retrievers = {}

    if not BASE_DOC_FOLDER.exists():
        print("⚠️ legal_docs folder missing")
        return retrievers

    for folder in BASE_DOC_FOLDER.iterdir():
        if folder.is_dir():
            retrievers[folder.name] = build_domain_index(folder)

    print("\n✅ All retrievers ready!")
    return retrievers


retrievers = load_all_retrievers()
