from rag import load_documents

print("Starting document indexing...")

count = load_documents()

print(f"Indexed {count} chunks successfully.")