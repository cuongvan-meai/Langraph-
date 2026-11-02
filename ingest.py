"""
ingest.py - CHẠY 1 LẦN: đưa roadmap.md vào ChromaDB.
Flow: đọc file -> chunk -> embed (nomic-embed-text) -> lưu ChromaDB
Chạy lại khi bạn sửa roadmap.md (script tự xóa dữ liệu cũ).
"""
import chromadb
import ollama

from tools import COLLECTION, DB_PATH, EMBED_MODEL, chunk_roadmap


def main():
    with open("roadmap.md", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_roadmap(text)
    print(f"Số chunk: {len(chunks)}")

    embeddings = ollama.embed(model=EMBED_MODEL, input=chunks)["embeddings"]
    print(f"Số vector: {len(embeddings)} | Số chiều: {len(embeddings[0])}")

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)  # xóa bản cũ để khỏi trùng lặp
    except Exception:
        pass
    col = client.create_collection(COLLECTION)
    col.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk{i}" for i in range(len(chunks))],
    )
    print("Xong! Đã lưu vào ChromaDB.")


if __name__ == "__main__":
    main()
