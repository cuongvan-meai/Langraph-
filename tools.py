"""
tools.py - 3 CÔNG CỤ (tool) của Agent
Mỗi tool là 1 hàm Python bình thường. Phần docstring RẤT QUAN TRỌNG:
LLM đọc docstring để biết khi nào nên dùng tool nào (bạn đã học ở Function Calling).
"""
import ast
import json
import operator
import re

import chromadb
import ollama

EMBED_MODEL = "nomic-embed-text"
DB_PATH = "./chroma_db"
COLLECTION = "lo_trinh"


# ---------- Chunking (cải tiến từ chunk_text bạn đã viết) ----------
def chunk_text(text, chunk_size=500, overlap=100):
    """Cắt text thành các đoạn chồng lấp nhau (giống hàm bạn đã tự viết)."""
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_roadmap(text):
    """
    Cắt theo từng mục '## ' (mỗi giai đoạn là 1 mục) TRƯỚC,
    mục nào dài quá mới cắt tiếp bằng chunk_text.
    Cách này tránh lỗi 'Embedding l' bị cắt đôi giữa từ mà bạn gặp lúc trước.
    Mỗi chunk đều giữ tiêu đề giai đoạn để khi tìm ra vẫn biết nó thuộc giai đoạn nào.
    """
    sections = re.split(r"\n(?=## )", text)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        title = sec.split("\n")[0]
        if len(sec) <= 700:
            chunks.append(sec)
        else:
            for i, part in enumerate(chunk_text(sec, 600, 100)):
                chunks.append(part if i == 0 else f"{title} (tiếp)\n{part}")
    return chunks


# ---------- Tool 1: tìm trong lộ trình (RAG) ----------
def search_docs(query: str) -> str:
    """Tìm thông tin trong tài liệu LỘ TRÌNH học AI Engineer: nội dung từng giai đoạn,
    kỹ năng cần học, mức độ % cần đạt, bài tập. Dùng khi câu hỏi hỏi về NỘI DUNG lộ trình.

    Args:
        query: câu hỏi hoặc từ khóa cần tìm, ví dụ "Giai đoạn 7 học gì"
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        col = client.get_collection(COLLECTION)
    except Exception:
        return "Lỗi: chưa có dữ liệu. Hãy chạy 'python ingest.py' trước."
    q_emb = ollama.embed(model=EMBED_MODEL, input=[query])["embeddings"]
    res = col.query(query_embeddings=q_emb, n_results=3)
    docs = res["documents"][0]
    return "\n\n---\n\n".join(docs) if docs else "Không tìm thấy thông tin liên quan."


# ---------- Tool 2: đọc tiến độ ----------
def get_progress() -> str:
    """Đọc TIẾN ĐỘ học của người dùng: đang ở giai đoạn mấy, đã hoàn thành những giai đoạn nào.
    Dùng khi hỏi 'mình đang ở đâu', 'đã học xong gì', hoặc khi cần biết vị trí hiện tại
    để gợi ý nên học gì tiếp. Không cần tham số.
    """
    try:
        with open("tien_do.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return "Lỗi: không tìm thấy file tien_do.json"
    except json.JSONDecodeError:
        return "Lỗi: file tien_do.json sai định dạng JSON"
    return json.dumps(data, ensure_ascii=False)


# ---------- Tool 3: máy tính an toàn ----------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("biểu thức không hợp lệ")


def calculator(expression: str) -> str:
    """Tính một biểu thức toán học, ví dụ '6/10*100' hoặc '25*48'.
    Dùng khi cần tính phần trăm, tổng, hiệu, tích, thương.

    Args:
        expression: biểu thức toán, chỉ gồm số và + - * / % **
    """
    try:
        # Dùng ast thay cho eval() để tránh chạy code nguy hiểm
        result = _eval(ast.parse(expression.strip(), mode="eval").body)
        return str(round(result, 4))
    except Exception as e:
        return f"Lỗi tính toán: {e}"


# Danh sách để đưa cho LLM + bảng tra tên -> hàm để thực thi
TOOLS = [search_docs, get_progress, calculator]
TOOL_MAP = {t.__name__: t for t in TOOLS}
