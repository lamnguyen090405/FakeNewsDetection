import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
import contextlib
import re
import unicodedata
from underthesea import word_tokenize
import os

# --- CẤU HÌNH ---
STOPWORDS_FILE = 'vietnamese-stopwords.txt'
MODEL_PATH = 'vinai/phobert-base' # Load Online từ HuggingFace

# --- 1. ĐỊNH NGHĨA DỮ LIỆU ĐẦU VÀO ---
class NewsInput(BaseModel):
    source: str
    title: str
    content: str

# --- 2. BIẾN TOÀN CỤC (Để lưu model đã load) ---
ml_models = {}
STOPWORDS = set()

# --- 3. CÁC HÀM HỖ TRỢ (HELPER FUNCTIONS) ---

def load_stopwords_global():
    """Đọc file stopwords và lưu vào bộ nhớ"""
    global STOPWORDS
    try:
        if os.path.exists(STOPWORDS_FILE):
            with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        STOPWORDS.add(word)
                        # Thêm cả biến thể có gạch dưới (cho underthesea)
                        STOPWORDS.add(word.replace(' ', '_'))
            print(f"✅ Đã tải {len(STOPWORDS)} từ dừng.")
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy file '{STOPWORDS_FILE}'. Bỏ qua bước lọc stopword.")
    except Exception as e:
        print(f"❌ Lỗi khi đọc stopwords: {e}")

def text_preprocess(text):
    """
    Quy trình làm sạch dữ liệu (Phải khớp 100% với lúc Train)
    1. Unicode -> 2. Regex -> 3. Lower -> 4. Tokenize -> 5. Stopwords
    """
    if not isinstance(text, str) or text is None:
        return ""

    # B1: Chuẩn hóa Unicode
    text = unicodedata.normalize('NFC', text)

    # B2: Regex làm sạch (Bỏ HTML, URL, Email, Ký tự đặc biệt, Số)
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\S*@\S*\s?', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    # B3: Chuyển thường & Bỏ khoảng trắng thừa
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    # B4: Tách từ (Tokenize) bằng Underthesea
    try:
        text = word_tokenize(text, format='text')
    except:
        pass # Nếu lỗi thì giữ nguyên text

    # B5: Loại bỏ Stopwords
    if text and STOPWORDS:
        words = text.split()
        clean_words = [w for w in words if w not in STOPWORDS]
        text = " ".join(clean_words)

    return text

def get_embeddings(text_list, max_len=256):
    """Chuyển đổi text thành vector bằng PhoBERT"""
    tokenizer = ml_models['tokenizer']
    bert = ml_models['bert']
    device = ml_models['device']
    
    inputs = tokenizer(
        text_list, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=max_len
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = bert(**inputs)
    
    # Lấy vector CLS (token đầu tiên)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# --- 4. LIFESPAN (QUẢN LÝ KHỞI ĐỘNG SERVER) ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n⏳ Đang khởi động Server & Tải Models...")
    
    # 1. Load Stopwords
    load_stopwords_global()

    # 2. Load Scikit-learn Models
    try:
        ml_models['le'] = joblib.load('models/label_encoder.pkl')
        ml_models['mlp'] = joblib.load('models/news_classifier_mlp.pkl')
        print("✅ Đã tải MLP Model & Label Encoder.")
    except Exception as e:
        print(f"❌ Lỗi tải file .pkl: {e}")
        print("💡 Gợi ý: Kiểm tra xem file đã nằm trong thư mục 'models/' chưa?")

    # 3. Load PhoBERT (Online)
    try:
        print(f"⏳ Đang tải PhoBERT từ {MODEL_PATH} (có thể mất 30s)...")
        ml_models['tokenizer'] = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
        ml_models['bert'] = AutoModel.from_pretrained(MODEL_PATH)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ml_models['bert'].to(device)
        ml_models['device'] = device
        print(f"✅ Đã tải xong PhoBERT (Device: {device})")
    except Exception as e:
        print(f"❌ Lỗi tải PhoBERT: {e}")

    print("🚀 Server đã sẵn sàng!\n")
    yield
    # Dọn dẹp khi tắt
    ml_models.clear()

# --- 5. KHỞI TẠO APP & ROUTES ---
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Trả về giao diện HTML"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(data: NewsInput):
    """API Dự đoán chính"""
    try:
        # 1. Tiền xử lý dữ liệu (Pre-processing)
        # Quan trọng: Phải làm sạch Title và Content trước khi đưa vào PhoBERT
        clean_title = text_preprocess(data.title)
        clean_content = text_preprocess(data.content)

        # 2. Feature Engineering
        # Token count: Tính trên nội dung gốc (theo logic manual test cũ)
        token_count = len(data.content.split())
        
        # Embeddings: Tính trên nội dung ĐÃ LÀM SẠCH
        title_emb = get_embeddings([clean_title], max_len=64)
        content_emb = get_embeddings([clean_content], max_len=256)
        
        # Source Hash: Logic cũ
        source_feat = np.array([[hash(data.source) % 1000]])
        token_feat = np.array([[token_count]])
        
        # 3. Gộp features (Stacking)
        full_features = np.hstack([title_emb, content_emb, source_feat, token_feat])
        
        # 4. Dự đoán (Inference)
        mlp = ml_models['mlp']
        le = ml_models['le']
        
        pred_idx = mlp.predict(full_features)[0]
        label = le.inverse_transform([pred_idx])[0]
        
        # Tính xác suất tin cậy (Confidence score)
        pred_proba = mlp.predict_proba(full_features).max()
        
        return {
            "status": "success",
            "prediction": label,
            "confidence": f"{pred_proba*100:.2f}%"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- 6. CHẠY SERVER ---
if __name__ == '__main__':
    # Lưu ý: Khi chạy file này trực tiếp, không dùng reload để tránh lỗi Windows
    uvicorn.run(app, host="127.0.0.1", port=8000)