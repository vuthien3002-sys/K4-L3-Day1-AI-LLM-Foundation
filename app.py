import streamlit as st
import pandas as pd
import os
from openai import OpenAI

# Nạp trực tiếp các biến và hàm từ file bài tập của bạn
from template import (
    OPENAI_MODEL, OPENAI_MINI_MODEL,
    compare_models, batch_compare, estimate_cost
)

# Cấu hình giao diện toàn màn hình
st.set_page_config(page_title="LLM API Dashboard", layout="wide")
st.title("chat bot")

# Phân chia 3 không gian làm việc
tab1, tab2, tab3 = st.tabs(["💬 Trợ lý Chat (Streaming)", "⚖️ So sánh Model", "📊 Phân tích Hàng loạt"])

# ==========================================
# TAB 1: CHATBOT CÓ PERSONA VÀ TÍNH PHÍ
# ==========================================
with tab1:
    st.header("Kiểm thử Persona & Streaming")
    
    # Cho phép người dùng tùy chỉnh System Prompt trực tiếp trên UI
    persona = st.text_input("Định hình nhân vật (System Prompt):", "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.")

    # Khởi tạo bộ nhớ cho session chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # In lại lịch sử chat trên màn hình
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Khung nhập liệu
    if prompt := st.chat_input("Nhập tin nhắn..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Khởi tạo client OpenAI (Lấy cấu hình môi trường giống trong template.py)
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
            )
            
            # Gắn persona vào đầu, kết hợp 6 tin nhắn gần nhất (3 lượt)
            messages = [{"role": "system", "content": persona}] + st.session_state.messages[-6:]
            
            # Gọi streaming
            stream = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True
            )
            response = st.write_stream(stream)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Gọi hàm estimate_cost từ template.py để tính tiền cho lượt chat vừa rồi
        cost_info = estimate_cost(prompt, response, OPENAI_MODEL)
        st.caption(f"**Thống kê lượt gọi:** {cost_info['input_tokens']} token vào | {cost_info['output_tokens']} token ra | Chi phí: `{cost_info['total_cost']:.6f}$`")

# ==========================================
# TAB 2: SO SÁNH ĐƠN LẺ
# ==========================================
with tab2:
    st.header("So sánh tốc độ và chi phí: Model Lớn vs Mini")
    single_prompt = st.text_area("Nhập câu lệnh cần kiểm tra:", "Giải thích cơ chế hoạt động của blockchain trong 3 câu.")
    
    if st.button("So sánh ngay"):
        with st.spinner("Đang gọi API..."):
            res = compare_models(single_prompt)
            
            # Chia 2 cột để hiển thị kết quả song song
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"Mô hình Lớn ({OPENAI_MODEL})")
                st.write(res['gpt4o_response'])
                st.caption(f"⏱️ Thời gian: {res['gpt4o_latency']:.2f}s | 💰 Chi phí ước tính: {res['gpt4o_cost_estimate']:.6f}$")
                
            with c2:
                st.info(f"Mô hình Mini ({OPENAI_MINI_MODEL})")
                st.write(res['mini_response'])
                st.caption(f"⏱️ Thời gian: {res['mini_latency']:.2f}s")

# ==========================================
# TAB 3: CHẠY HÀNG LOẠT (BONUS)
# ==========================================
with tab3:
    st.header("So sánh hiệu suất hàng loạt (Batch)")
    batch_input = st.text_area(
        "Nhập danh sách câu lệnh (mỗi câu 1 dòng):", 
        "Ưu điểm của Python là gì?\nViết 1 vòng lặp for trong C++\nBầu trời tại sao màu xanh?"
    )
    
    if st.button("Chạy Batch Test"):
        # Tách chuỗi thành list các prompt
        prompts = [p.strip() for p in batch_input.split('\n') if p.strip()]
        
        if prompts:
            with st.spinner(f"Đang xử lý {len(prompts)} câu lệnh..."):
                batch_res = batch_compare(prompts)
                
                # Chuyển đổi list dictionary thành bảng Pandas để hiển thị đẹp mắt trên Streamlit
                df = pd.DataFrame(batch_res)
                df = df[['prompt', 'gpt4o_latency', 'mini_latency', 'gpt4o_cost_estimate', 'gpt4o_response', 'mini_response']]
                
                # Hiển thị bảng dữ liệu tương tác
                st.dataframe(df, use_container_width=True)
        else:
            st.warning("Vui lòng nhập ít nhất 1 câu lệnh.")