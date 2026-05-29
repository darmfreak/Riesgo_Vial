"""
build_index.py — Construye el índice FAISS para el RAG de RiesgoVial.
Ejecutar desde riesgo_api/:  python rag/build_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_index(city: str = "medellin"):
    docs_path = Path(__file__).parent / "documents" / city
    idx_path  = Path(__file__).parent / "index" / city

    if not docs_path.exists():
        print(f"❌ No existe: {docs_path}")
        return

    loader = DirectoryLoader(str(docs_path), glob="*.txt", loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    print(f"📄 {len(docs)} documentos cargados")

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    chunks = splitter.split_documents(docs)
    print(f"🔪 {len(chunks)} chunks generados")

    print("⚙️  Cargando modelo de embeddings…")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )

    db = FAISS.from_documents(chunks, embeddings)
    idx_path.mkdir(parents=True, exist_ok=True)
    db.save_local(str(idx_path))
    print(f"✅ Índice FAISS guardado: {idx_path}  ({len(chunks)} chunks)")


if __name__ == "__main__":
    build_index("medellin")
