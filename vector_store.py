import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

PDF_FOLDER = r"D:\Projects\LegalAi\legal_docs"
DB_LOCATION = "./chroma_db"


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

add_documents = not os.path.exists(DB_LOCATION)

if add_documents:
    documents = []

    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
            pages = loader.load()

            for page in pages:
                # Add metadata for filtering + citation
                page.metadata["source"] = file
                page.metadata["page"] = page.metadata.get("page", 0)

                documents.append(page)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = text_splitter.split_documents(documents)

# Create vector store
vector_store = Chroma(
    collection_name="legal_documents",
    persist_directory=DB_LOCATION,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(split_docs)

# 🔥 Use MMR Retrieval
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 20
    }
)

# 🔥 OPTIONAL FILTER VERSION (use when needed)
def get_filtered_retriever(filename):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "filter": {"source": filename}
        }
    )
