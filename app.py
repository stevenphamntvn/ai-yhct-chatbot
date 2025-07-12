# file: app.py
# Phiên bản hoàn chỉnh: Chạy online, sửa lỗi, và cho phép lựa chọn mô hình AI.

# --- PHẦN SỬA LỖI QUAN TRỌNG CHO STREAMLIT CLOUD ---
# Ba dòng này phải nằm ở ngay đầu file
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ----------------------------------------------------

import streamlit as st
import chromadb
import google.generativeai as genai
import os
import requests
import zipfile
from io import BytesIO

# --- PHẦN CẤU HÌNH ---
# API Key của bạn từ Google Cloud
GOOGLE_API_KEY = 'AIzaSyBOAgpJI1voNNxeOC6sS7y01EJRXWSK0YU' # !!! THAY API KEY CỦA BẠN VÀO ĐÂY !!!

# --- CẤU HÌNH TRIỂN KHAI ONLINE ---
# !!! QUAN TRỌNG: Dán đường dẫn tải trực tiếp file zip của bạn vào đây
DB_ZIP_URL = "https://drive.google.com/uc?export=download&id=1-2q9AG84492czMsWmhTbQziBDRyvFP0X"
DB_PATH = 'yhct_chroma_db'
COLLECTION_NAME = 'yhct_collection'

# --- BẢNG GIÁ VÀ LỰA CHỌN MÔ HÌNH ---
MODEL_PRICING = {
    "gemini-1.5-flash-latest": {
        "input": 0.35,  # $0.35 cho mỗi 1 triệu token
        "output": 1.05  # $1.05 cho mỗi 1 triệu token
    },
    "gemini-1.5-pro-latest": {
        "input": 3.50,  # $3.50 cho mỗi 1 triệu token
        "output": 10.50 # $10.50 cho mỗi 1 triệu token
    }
}
MODEL_OPTIONS = list(MODEL_PRICING.keys())

# --- HÀM TẢI VÀ GIẢI NÉN DATABASE ---
def setup_database():
    """Kiểm tra, tải về và giải nén database nếu cần."""
    if not os.path.exists(DB_PATH):
        st.info(f"Không tìm thấy database '{DB_PATH}'. Bắt đầu tải về từ cloud...")
        st.warning("Quá trình này chỉ diễn ra một lần và có thể mất vài phút.")
        
        if not DB_ZIP_URL or DB_ZIP_URL == "YOUR_DIRECT_DOWNLOAD_LINK_TO_THE_DB_ZIP_FILE":
            st.error("Lỗi cấu hình: Vui lòng cung cấp DB_ZIP_URL trong file app.py.")
            return False

        try:
            with st.spinner('Đang tải database...'):
                response = requests.get(DB_ZIP_URL)
                response.raise_for_status()
            with st.spinner('Đang giải nén...'):
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    z.extractall('.')
            st.success("Thiết lập database thành công! Đang tải lại...")
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi khi tải hoặc giải nén database: {e}")
            return False
    return True

# --- KHỞI TẠO DATABASE ---
@st.cache_resource
def load_db():
    """Kết nối tới database và trả về collection."""
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection
    except Exception as e:
        st.error(f"Lỗi kết nối database: {e}")
        return None

# --- HÀM LOGIC XỬ LÝ CÂU HỎI ---
def get_ai_response(question, model, collection, model_name):
    """Lấy câu trả lời từ AI và thông tin sử dụng."""
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n\n---\n\n".join(results['documents'][0])
    prompt = f"""Dựa vào thông tin tham khảo dưới đây, hãy trả lời câu hỏi của người dùng.
    Thông tin tham khảo: {context}
    Câu hỏi: {question}"""
    
    response = model.generate_content(prompt)
    usage_info = None
    try:
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count
        response_tokens = usage.candidates_token_count
        
        price_input = MODEL_PRICING[model_name]["input"]
        price_output = MODEL_PRICING[model_name]["output"]
        
        input_cost = (prompt_tokens / 1_000_000) * price_input
        output_cost = (response_tokens / 1_000_000) * price_output
        
        usage_info = {
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": usage.total_token_count,
            "cost_usd": input_cost + output_cost
        }
    except Exception:
        pass # Bỏ qua nếu không lấy được thông tin sử dụng

    return response.text, set(meta['source'] for meta in results['metadatas'][0]), usage_info

# --- GIAO DIỆN NGƯỜI DÙNG STREAMLIT ---
st.set_page_config(page_title="Trợ lý Y học Cổ truyền", page_icon="🌿")
st.title("🌿 Trợ lý Y học Cổ truyền")

# Thanh bên để chọn mô hình
with st.sidebar:
    st.header("Cấu hình")
    selected_model = st.selectbox(
        "Chọn mô hình AI:",
        options=MODEL_OPTIONS,
        index=0, # Mặc định chọn 'gemini-1.5-flash-latest'
        help="Flash nhanh và rẻ hơn, Pro thông minh hơn."
    )
    st.caption(f"Bạn đã chọn: `{selected_model}`")

# Bước 1: Đảm bảo database đã sẵn sàng
if setup_database():
    # Bước 2: Khởi tạo AI và DB
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        llm_model = genai.GenerativeModel(selected_model)
        collection = load_db()
    except Exception as e:
        st.error(f"Lỗi khởi tạo AI. Vui lòng kiểm tra API Key. Lỗi: {e}")
        llm_model = None
        collection = None

    # Bước 3: Hiển thị giao diện chat
    if llm_model and collection:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

        if prompt := st.chat_input("Ví dụ: Bệnh Thái Dương là gì?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner(f"AI ({selected_model}) đang suy nghĩ..."):
                    response_text, sources, usage_info = get_ai_response(prompt, llm_model, collection, selected_model)
                    
                    source_markdown = "\n\n---\n**Nguồn tham khảo:**\n" + "\n".join([f"- `{s}`" for s in sources])
                    full_response_text = response_text + source_markdown
                    st.markdown(full_response_text)
                    
                    if usage_info:
                        with st.expander("Xem chi tiết sử dụng API"):
                            st.metric("Chi phí (USD)", f"${usage_info['cost_usd']:.6f}")
                            st.caption(f"Tokens: {usage_info['total_tokens']} | Đầu vào: {usage_info['prompt_tokens']} | Đầu ra: {usage_info['response_tokens']}")

            # Lưu lại vào lịch sử chat
            if usage_info:
                usage_html = f"""
                <details>
                    <summary>Xem chi tiết sử dụng API</summary>
                    <p><b>Chi phí (USD):</b> ${usage_info['cost_usd']:.6f}</p>
                </details>
                """
                full_response_to_save = full_response_text + usage_html
            else:
                full_response_to_save = full_response_text
            st.session_state.messages.append({"role": "assistant", "content": full_response_to_save})
