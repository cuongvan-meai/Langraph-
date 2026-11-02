"""
agent.py - AGENT LANGGRAPH: trợ lý hỏi đáp về lộ trình AI Engineer

3 khái niệm LangGraph (đọc kỹ phần này):
  STATE : "cái hộp dữ liệu chung" chạy xuyên suốt graph. Ở đây chứa danh sách tin nhắn.
  NODE  : 1 bước xử lý = 1 hàm nhận State, trả về phần State cần cập nhật.
  EDGE  : đường nối giữa các node. Edge CÓ ĐIỀU KIỆN = hàm quyết định đi nhánh nào.

Sơ đồ:
                 ┌───────────────┐
   START ──────► │   llm (node)  │ ◄────────────┐
                 └───────┬───────┘              │
              có tool_calls?                    │
              ┌──────────┴──────────┐           │
            CÓ                    KHÔNG         │
              ▼                     ▼           │
        ┌───────────┐              END           │
        │ tools     │ ───────────────────────────┘
        │ (node)    │   chạy tool xong quay lại llm
        └───────────┘

Đây chính là ReAct loop bạn đã học: llm = Thought/Action, tools = Observation,
nhưng có cấu trúc rõ ràng và có giới hạn số bước (không loop vô tận như while True).
"""
import re
from typing import TypedDict

import ollama
from langgraph.graph import END, START, StateGraph

from tools import TOOL_MAP, TOOLS

MODEL = "qwen3:4b"
MAX_STEPS = 6  # chặn loop vô tận: tối đa 6 lần gọi LLM cho 1 câu hỏi

SYSTEM_PROMPT = """Bạn là trợ lý học tập, giúp người dùng về LỘ TRÌNH học AI Engineer của họ.
Luôn trả lời bằng tiếng Việt, ngắn gọn, dễ hiểu.

Bạn có 3 công cụ:
- search_docs: tra NỘI DUNG lộ trình (giai đoạn nào học gì, mức % cần đạt, bài tập).
- get_progress: xem TIẾN ĐỘ của người dùng (đang ở giai đoạn mấy, đã xong gì).
- calculator: tính toán (ví dụ phần trăm hoàn thành).

Quy tắc:
1. Chào hỏi, trò chuyện thông thường: trả lời thẳng, KHÔNG gọi công cụ.
2. Hỏi nội dung lộ trình: gọi search_docs.
3. Hỏi "đang ở đâu / đã học gì": gọi get_progress.
4. Hỏi "còn bao nhiêu %": gọi get_progress rồi calculator.
5. Hỏi "nên học gì tiếp": gọi get_progress trước, rồi search_docs cho giai đoạn hiện tại.
6. Chỉ trả lời dựa trên kết quả công cụ. Không có thông tin thì nói "không có trong lộ trình", đừng bịa.
"""


# ---------- STATE ----------
class State(TypedDict):
    messages: list  # toàn bộ hội thoại: system, user, assistant, tool
    steps: int      # đã gọi LLM bao nhiêu lần (để chặn loop)


# ---------- Hàm gọi LLM (tách riêng để dễ test) ----------
def call_llm(messages, use_tools=True):
    kwargs = {"model": MODEL, "messages": messages}
    if use_tools:
        kwargs["tools"] = TOOLS  # ollama tự đọc docstring của hàm để tạo tool schema
    try:
        return ollama.chat(think=False, **kwargs)  # tắt chế độ "suy nghĩ" của qwen3 cho nhanh
    except TypeError:
        return ollama.chat(**kwargs)               # bản ollama cũ không có tham số think


def clean(text):
    """Xóa phần <think>...</think> nếu model vẫn xuất ra."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


# ---------- NODE 1: llm ----------
def llm_node(state: State):
    steps = state["steps"]
    # Quá số bước cho phép -> ép LLM trả lời luôn, không cho gọi tool nữa
    resp = call_llm(state["messages"], use_tools=steps < MAX_STEPS)
    msg = resp.message

    new_msg = {"role": "assistant", "content": clean(msg.content)}
    if msg.tool_calls:
        new_msg["tool_calls"] = [
            {"function": {"name": tc.function.name, "arguments": dict(tc.function.arguments)}}
            for tc in msg.tool_calls
        ]
    # Trả về phần State cần CẬP NHẬT (LangGraph tự gộp vào State hiện tại)
    return {"messages": state["messages"] + [new_msg], "steps": steps + 1}


# ---------- NODE 2: tools ----------
def tools_node(state: State):
    last = state["messages"][-1]
    results = []
    for tc in last["tool_calls"]:
        name = tc["function"]["name"]
        args = tc["function"]["arguments"]
        func = TOOL_MAP.get(name)
        if func is None:  # LLM bịa ra tool không tồn tại
            output = f"Lỗi: không có công cụ tên '{name}'"
        else:
            try:
                output = func(**args)
            except Exception as e:  # LLM truyền sai tham số
                output = f"Lỗi khi chạy {name}: {e}"
        print(f"   🔧 {name}({args}) -> {str(output)[:80]}...")
        results.append({"role": "tool", "tool_name": name, "content": str(output)})
    return {"messages": state["messages"] + results}


# ---------- EDGE CÓ ĐIỀU KIỆN ----------
def should_continue(state: State):
    """Sau node llm: LLM muốn gọi tool -> sang 'tools'. Không -> kết thúc."""
    if state["messages"][-1].get("tool_calls"):
        return "tools"
    return END


# ---------- LẮP RÁP GRAPH ----------
def build_graph():
    g = StateGraph(State)
    g.add_node("llm", llm_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "llm")                              # bắt đầu -> llm
    g.add_conditional_edges("llm", should_continue,       # llm -> (tools | END)
                            {"tools": "tools", END: END})
    g.add_edge("tools", "llm")                            # tools xong -> quay lại llm
    return g.compile()


def ask(graph, question):
    state = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "steps": 0,
    }
    final = graph.invoke(state)
    return final["messages"][-1]["content"]


if __name__ == "__main__":
    graph = build_graph()
    print("Agent lộ trình AI Engineer (gõ 'thoat' để thoát)\n")
    while True:
        q = input("Bạn: ").strip()
        if q.lower() in ("thoat", "exit", "quit"):
            break
        if q:
            print(f"Agent: {ask(graph, q)}\n")
