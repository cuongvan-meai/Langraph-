"""
test_offline.py - Kiểm tra LOGIC không cần Ollama (dùng LLM và embedding giả).
Chạy: python test_offline.py
Nếu tất cả ✅ thì graph, tool, chunking đều đúng; còn lại chỉ phụ thuộc chất lượng qwen3:4b.
"""
import hashlib
import shutil
from types import SimpleNamespace as NS

import ollama

import agent
import ingest
import tools


# --- Embedding giả: băm từng từ vào vector 64 chiều (đủ để test tìm kiếm theo từ khóa) ---
def fake_embed(model, input):
    vecs = []
    for text in input:
        v = [0.0] * 64
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        vecs.append(v)
    return {"embeddings": vecs}


ollama.embed = fake_embed
tools.DB_PATH = "./chroma_test"
ingest.DB_PATH = "./chroma_test"
shutil.rmtree("./chroma_test", ignore_errors=True)

# 1) chunking + ingest
chunks = tools.chunk_roadmap(open("roadmap.md", encoding="utf-8").read())
assert len(chunks) >= 10, "chunk quá ít"
assert all(c.startswith("#") or "(tiếp)" in c or len(c) > 0 for c in chunks)
ingest.main()
print("✅ ingest OK:", len(chunks), "chunks")

# 2) search_docs
out = tools.search_docs("Giai đoạn 7 Fine-tuning LoRA QLoRA")
assert "LoRA" in out, out
print("✅ search_docs OK")

# 3) get_progress + calculator
assert '"giai_doan_hien_tai": 6' in tools.get_progress()
assert tools.calculator("6/10*100") == "60.0"
assert "Lỗi" in tools.calculator("__import__('os').system('ls')")  # chặn code nguy hiểm
print("✅ get_progress + calculator OK (và chặn được code độc)")


# --- LLM giả theo kịch bản ---
def tc(name, **args):
    return NS(function=NS(name=name, arguments=args))


def make_llm(script):
    it = iter(script)

    def fake(messages, use_tools=True):
        content, calls = next(it)
        return NS(message=NS(content=content, tool_calls=calls))
    return fake


graph = agent.build_graph()

# 4) Chào hỏi: không gọi tool
agent.call_llm = make_llm([("Chào bạn! Mình giúp gì được?", None)])
assert "Chào" in agent.ask(graph, "Chào bạn")
print("✅ 'Chào bạn' -> không gọi tool")

# 5) Còn bao nhiêu %: get_progress -> calculator -> trả lời
agent.call_llm = make_llm([
    ("", [tc("get_progress")]),
    ("", [tc("calculator", expression="6/10*100")]),
    ("Bạn đã hoàn thành 60% lộ trình.", None),
])
r = agent.ask(graph, "Còn bao nhiêu % nữa là xong?")
assert "60" in r
print("✅ chuỗi get_progress -> calculator -> trả lời")

# 6) LLM gọi tool bịa -> không crash
agent.call_llm = make_llm([("", [tc("tool_bia")]), ("Xin lỗi, mình không làm được.", None)])
agent.ask(graph, "test")
print("✅ tool không tồn tại -> xử lý lỗi, không crash")

# 7) Chặn loop vô tận: LLM cứ đòi gọi tool mãi
calls = {"n": 0}


def looping(messages, use_tools=True):
    calls["n"] += 1
    if use_tools:
        return NS(message=NS(content="", tool_calls=[tc("get_progress")]))
    return NS(message=NS(content="Đây là câu trả lời cuối.", tool_calls=None))


agent.call_llm = looping
r = agent.ask(graph, "loop")
assert calls["n"] == agent.MAX_STEPS + 1 and "cuối" in r
print(f"✅ chặn loop: dừng sau {calls['n']} lần gọi LLM")

shutil.rmtree("./chroma_test", ignore_errors=True)
print("\n🎉 Tất cả test logic đều qua!")
