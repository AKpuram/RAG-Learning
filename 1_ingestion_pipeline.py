
import os

from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Configuration
DOCS_PATH = "docs"
PERSISTENT_DIRECTORY = "db/chroma_db"
COLLECTION_NAME = "company_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def documents_load(docs_path=DOCS_PATH):
    """Load all .txt files from the documents directory."""

    print(f"Loading documents from {docs_path}...")

    if not os.path.isdir(docs_path):
        raise FileNotFoundError(
            f"Directory '{docs_path}' does not exist."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()

    if not documents:
        raise FileNotFoundError(
            f"No .txt files found in '{docs_path}'. "
            "Please add your company documents."
        )

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into smaller chunks for RAG."""

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No chunks were created. Check the document contents."
        )

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
        print(f"Length: {len(chunk.page_content)} characters")
        print(chunk.page_content)
        print("-" * 50)

    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")

    print(f"\nTotal chunks created: {len(chunks)}")
    return chunks


def create_embeddings():
    """Create the embedding model used by ChromaDB."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def create_embeddings_and_store_in_vector_database(
    chunks,
    persist_directory=PERSISTENT_DIRECTORY,
):
    """Create embeddings and store chunks in ChromaDB."""

    os.makedirs(persist_directory, exist_ok=True)

    embedding_model = create_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print("\nFinished creating vector store.")
    print(f"Vector store saved to: {persist_directory}")
    print(f"Stored chunks: {vectorstore._collection.count()}")

    return vectorstore


def main():
    """Main document ingestion pipeline."""

    database_file = os.path.join(
        PERSISTENT_DIRECTORY,
        "chroma.sqlite3",
    )

    # Reuse the database if it already exists.
    if os.path.isfile(database_file):
        print("Existing vector store found. Loading it...")

        embedding_model = create_embeddings()

        vectorstore = Chroma(
            persist_directory=PERSISTENT_DIRECTORY,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
        )

        print(
            "Loaded vector store with "
            f"{vectorstore._collection.count()} stored chunks."
        )

        return vectorstore

    print("Vector store not found. Starting ingestion...\n")

    documents = documents_load(DOCS_PATH)
    chunks = split_documents(documents)

    vectorstore = create_embeddings_and_store_in_vector_database(
        chunks,
        PERSISTENT_DIRECTORY,
    )

    print("\nIngestion complete!")
    print("Your documents are ready for RAG queries.")

    return vectorstore


if __name__ == "__main__":
    main()
