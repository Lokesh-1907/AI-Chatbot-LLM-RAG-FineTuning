from pathlib import Path
import pymupdf
import chromadb
import requests

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_or_create_collection(
    name="documents"
)


def get_embedding(text):
    response = requests.post(
        "http://127.0.0.1:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["embedding"]


def split_text(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


def load_documents():
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return 0

    total_chunks = 0

    for pdf_file in pdf_files:

        print(f"Processing: {pdf_file.name}")

        pdf = pymupdf.open(pdf_file)

        full_text = ""

        for page in pdf:
            full_text += page.get_text()

        pdf.close()

        chunks = split_text(full_text)

        for index, chunk in enumerate(chunks):

            if not chunk.strip():
                continue

            embedding = get_embedding(chunk)

            collection.upsert(
                ids=[
                    f"{pdf_file.name}_{index}"
                ],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[
                    {
                        "source": pdf_file.name
                    }
                ]
            )

            total_chunks += 1

    return total_chunks


def search_documents(query, n_results=4):

    embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    return documents