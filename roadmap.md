# Lộ trình trở thành AI Engineer (LLM / GenAI / Agent)

**Xuất phát điểm:** DSA cơ bản + Python cơ bản
**Thời gian:** 10-20 giờ/tuần
**Ước tính tổng thời gian:** ~5-7 tháng
**Đã loại bỏ:** Toán nền tảng, Tiếng Anh (học just-in-time khi cần)

---

## Giải thích mức độ %

- **0-30%** — Biết khái niệm, hiểu cơ bản, đọc được code người khác
- **31-60%** — Tự viết được, dùng được trong project thực tế
- **61-80%** — Thành thạo, debug được, giải thích được cho người khác
- **81-100%** — Chuyên sâu (AI Engineer ứng dụng thường không cần lên đến đây)

---

## Giai đoạn 0 — Khởi động (1-2 tuần)

| Kỹ năng | Mức độ cần đạt | Lý do |
|---|---|---|
| Python OOP, file/JSON, try/except | 60% | Dùng hàng ngày khi code |
| Git & GitHub | 50% | Đủ để quản lý project cá nhân |
| Setup môi trường (VS Code, venv) | 60% | Nền tảng làm việc |
| Gọi API cơ bản | 50% | Bước đầu vào LLM |

---

## Giai đoạn 1 — Python cho AI & Data (2-3 tuần)

| Kỹ năng | Mức độ cần đạt | Lý do |
|---|---|---|
| NumPy | 50% | Đủ hiểu shape, array — không cần chuyên sâu |
| Pandas | 60% | Dùng nhiều khi xử lý data thực tế |
| Matplotlib / Seaborn | 40% | Vẽ được biểu đồ cơ bản là đủ |
| REST API & JSON | 60% | Gọi API LLM đều qua HTTP/JSON |

---

## Giai đoạn 2 — Nền tảng Machine Learning (3-4 tuần)

| Kỹ năng | Mức độ cần đạt | Lý do |
|---|---|---|
| Khái niệm ML (supervised/unsupervised) | 50% | Hiểu đủ để không bị lạc khi đọc tài liệu LLM |
| Scikit-learn | 50% | Dùng được, không cần thạo sâu |
| Đánh giá mô hình (accuracy, overfitting) | 60% | Quan trọng khi eval hệ thống AI sau này |
| XGBoost / Ensemble | 40% | Biết khái niệm, dùng được là đủ |

---

## Giai đoạn 3 — Deep Learning & Transformer (3-4 tuần)

| Kỹ năng | Mức độ cần đạt | Lý do |
|---|---|---|
| Mạng nơ-ron cơ bản | 50% | Hiểu đủ để không bị lạc khi đọc paper LLM |
| PyTorch cơ bản | 40% | Đọc được code, chạy được model có sẵn |
| Cơ chế Attention & Transformer | 60% | Nền tảng của mọi LLM — cần hiểu rõ |
| CNN / RNN | 20% | Chỉ cần biết khái niệm, không cần thực hành sâu |

---

## Giai đoạn 4 — LLM Foundations (3 tuần)

| Kỹ năng | Mức độ cần đạt | Lý do |
|---|---|---|
| Tokenization & Embeddings | 70% | Cốt lõi của mọi hệ thống LLM |
| Prompt Engineering | 80% | Kỹ năng dùng hàng ngày của AI Engineer |
| System Prompt Design | 80% | Quan trọng khi xây sản phẩm AI |
| Gọi API OpenAI / Anthropic | 70% | Công việc thực tế hàng ngày |
| Hugging Face / Ollama | 50% | Dùng được model local khi cần |

---

## Giai đoạn 5 — RAG & Vector Database (4 tuần)

| Kỹ năng | Mức độ cần đạt | Lý do |
|---|---|---|
| Kiến trúc RAG | 70% | Hệ thống AI thực tế đều dùng RAG |
| Chunking strategy | 60% | Ảnh hưởng trực tiếp đến chất lượng retrieval |
| Vector DB (Chroma/FAISS) | 60% | Công cụ cốt lõi của RAG |
| Reranking | 50% | Cải thiện chất lượng retrieval |
| LangChain / LlamaIndex | 60% | Framework phổ biến nhất hiện tại |

---

## Giai đoạn 6 — AI Agents (4 tuần)

| Kỹ năng | Mức độ cần đạt | Lý do |
|---|---|---|
| ReAct loop | 70% | Hiểu cơ chế agent hoạt động thế nào |
| Function calling / Tool use | 80% | Kỹ năng cốt lõi của AI Engineer 2026 |
| Structured outputs | 70% | Dùng trong mọi hệ thống agent thực tế |
| LangGraph / CrewAI / PydanticAI | 60% | Dùng được framework là đủ |
| Model Context Protocol (MCP) | 60% | Chuẩn mới đang được adopt rộng rãi |
| Multi-agent orchestration | 50% | Biết cách phối hợp nhiều agent |

---

## Giai đoạn 7 — Fine-tuning cơ bản (2-3 tuần)

| Kỹ năng | Mức độ cần đạt | Lý do |
|---|---|---|
| LoRA / QLoRA | 50% | Biết dùng, không cần hiểu sâu toán học |
| thư viện trl / peft | 50% | Chạy được fine-tuning là đủ |
| Chuẩn bị dataset | 60% | Data quality quyết định kết quả |

---

## Giai đoạn 8 — Evaluation, Observability & Deploy (3-4 tuần)

| Kỹ năng | Mức độ cần đạt | Lý do |
|---|---|---|
| RAGAS / Golden test set | 70% | Đánh giá hệ thống AI là kỹ năng quan trọng |
| Langfuse / LangSmith | 60% | Theo dõi và debug hệ thống production |
| FastAPI | 70% | Deploy model thành API |
| Docker cơ bản | 50% | Đóng gói ứng dụng |
| Phòng chống prompt injection | 70% | Bảo mật hệ thống AI |

---

## Giai đoạn 9 — Portfolio & Xin việc (4-6 tuần cuối)

| Kỹ năng | Mức độ cần đạt | Lý do |
|---|---|---|
| Viết README / Documentation | 70% | Nhà tuyển dụng đọc cái này đầu tiên |
| System design cho LLM | 60% | Hay bị hỏi trong phỏng vấn |
| Giải thích trade-off kiến trúc | 70% | Thể hiện tư duy kỹ sư |
| Demo / Presentation | 60% | Bán được project của mình |

---

### Ghi chú
- % là mức độ **cần đạt để đi tiếp**, không phải giới hạn — nếu bạn muốn đào sâu hơn hoàn toàn được
- Toán và tiếng Anh học just-in-time, không thành module riêng
- Ưu tiên thị trường 2026: LLM → RAG → Agent → Eval/Deploy
