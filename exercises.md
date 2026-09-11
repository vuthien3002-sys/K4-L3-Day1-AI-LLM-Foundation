# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Mức 0.0: Câu trả lời rất rập khuôn, an toàn, mang tính sự thật phổ biến (ví dụ: hang Sơn Đoòng, xe máy). Nếu chạy nhiều lần, kết quả gần như không đổi,Mức 0.5 và 1.0: Văn phong tự nhiên hơn, sáng tạo hơn, có thể kể đến những sự thật ngách hoặc thú vị hơn,Mức 1.5: Rất ngẫu nhiên. Mô hình có thể bắt đầu dùng từ ngữ lủng củng, lạc đề, hoặc thậm chí bịa đặt thông tin (hallucinate).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature ở mức rất thấp (từ 0.0 đến 0.2). Lý do là vì chatbot CSKH cần cung cấp thông tin chính xác, mang tính thực tế và nhất quán dựa trên tài liệu có sẵn của công ty, nghiêm cấm việc mô hình tự "sáng tạo" (hallucinate) hoặc bịa đặt các chính sách không có thật.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Ước tính: Chi phí sử dụng model lớn đắt hơn khoảng 16,67 lần so với bản mini cho cùng một lượng token đầu ra (75$/ngày so với 4.5$/ngày).

>Trường hợp model lớn xứng đáng: Các tác vụ đòi hỏi suy luận logic phức tạp, phân tích ngữ cảnh sâu hoặc viết code (ví dụ: gỡ lỗi chương trình điều khiển PLC, phân tích log an ninh mạng, hoặc đọc hiểu tài liệu kỹ thuật công nghiệp phức tạp).

>Trường hợp nên dùng mini: Các quy trình chạy tự động khối lượng lớn, cấu trúc đơn giản (ví dụ: trích xuất nhanh thông số từ văn bản, phân loại tự động các cảnh báo lỗi hệ thống, hoặc chatbot giao tiếp thông thường).

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phản hồi của "giáo viên tiểu học" thường ngắn gọn, dùng từ vựng cơ bản và ví dụ trực quan (như cuốn sổ dùng chung cho cả lớp). Ngược lại, "chuyên gia tài chính" đưa ra văn bản dài hơn, cấu trúc phức tạp và chứa nhiều thuật ngữ chuyên ngành (sổ cái phân tán, mã hóa, cơ chế đồng thuận). Điều này chứng minh System prompt đóng vai trò định hình bối cảnh cốt lõi; nó trực tiếp điều khiển "nhân cách", giọng văn và giới hạn chuyên môn của model trước khi nó bắt đầu trả lời câu hỏi của người dùng.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Trải nghiệm thực tế cho thấy số lượng token đếm bằng thư viện tiktoken thường cao hơn từ 50% đến hơn 100% so với công thức ước lượng thô (số từ / 0.75). Nguyên nhân là do các bộ Tokenizer hiện hành được huấn luyện và tối ưu hóa chủ yếu trên tập dữ liệu tiếng Anh. Tiếng Việt là ngôn ngữ đơn lập, dùng nhiều dấu thanh (diacritics), khiến Tokenizer thường xuyên bị "bối rối" và phải băm nhỏ một từ tiếng Việt thành nhiều token vụn vặt (thậm chí tách thành từng ký tự), làm đội chi phí token lên rất cao so với tiếng Anh có cùng độ dài ý nghĩa.
---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming (đổ chữ ra từ từ) quan trọng nhất trong các giao diện tương tác trực tiếp với người dùng (như màn hình chat), giúp giảm thời gian chờ đợi cảm nhận được (Time to First Token - TTFT), giữ người dùng không bị nản chí khi mô hình đang tạo các văn bản dài. Ngược lại, non-streaming (đợi toàn bộ kết quả rồi mới trả về một lần) sẽ phù hợp hơn cho các quy trình chạy ngầm (backend API), xử lý dữ liệu hàng loạt (batch processing), hoặc trích xuất dữ liệu có cấu trúc (như JSON), nơi hệ thống máy tính chỉ cần nhận kết quả hoàn chỉnh cuối cùng để xử lý bước tiếp theo chứ không cần hiệu ứng hiển thị thời gian thực.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp ngăn chặn hiện tượng "thundering herd" (hiệu ứng bầy đàn). Nếu hàng nghìn client cùng bị lỗi và đồng loạt gửi lại request sau đúng 1 giây (delay cố định), máy chủ vừa mới phục hồi sẽ lập tức bị quá tải và sập thêm lần nữa. Việc tăng dần thời gian chờ theo cấp số nhân giúp dàn trải lượng request retry ra nhiều thời điểm khác nhau, giảm áp lực cục bộ và cho phép hệ thống API có đủ không gian để tự phục hồi.
---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System Prompt đề xuất: "Bạn là một kỹ sư hệ thống lão luyện, chuyên hỗ trợ khắc phục sự cố phần mềm và tự động hóa. Trả lời cực kỳ ngắn gọn, đi thẳng vào nguyên nhân cốt lõi và luôn định dạng các đoạn mã hoặc thông số cấu hình trong block code."

>Giải thích lựa chọn: Yêu cầu "ngắn gọn, đi thẳng vào nguyên nhân" giúp tiết kiệm chi phí token và giảm tối đa độ trễ (latency). Yêu cầu "định dạng đoạn mã trong block code" ép model cấu trúc đầu ra gọn gàng, giúp người dùng dễ dàng copy-paste khi đang thao tác trực tiếp với các file cấu hình hoặc terminal.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: Trợ lý hiện tại quản lý ngữ cảnh quá thô sơ (chỉ giữ đúng 6 tin nhắn gần nhất bằng lệnh history[-6:]). Nếu quá trình gỡ lỗi hoặc trao đổi kéo dài, nó sẽ hoàn toàn quên mất bối cảnh ban đầu, khiến người dùng phải liên tục nhắc lại thông tin.

>Đề xuất cải thiện: Triển khai cơ chế "Sliding Window Summarization" (Tóm tắt trượt). Khi lịch sử hội thoại chuẩn bị vượt quá giới hạn token cho phép, hệ thống sẽ tự động gọi một luồng API ngầm để tóm tắt toàn bộ tin nhắn cũ thành một đoạn văn bản ngắn. Đoạn tóm tắt này sau đó được chèn vào đầu bối cảnh, giúp trợ lý duy trì trí nhớ dài hạn mà không làm "cháy" ngân sách token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
