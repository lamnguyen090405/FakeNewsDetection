import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from collections import Counter
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
MODEL_PATH = 'vinai/phobert-base'
MLP_MODEL_PATH = 'models/news_classifier_mlp_optimized.pkl' 
LE_MODEL_PATH = 'models/label_encoder.pkl'

# --- ĐỊNH NGHĨA DỮ LIỆU ---
class NewsInput(BaseModel):
    title: str
    content: str

# --- BIẾN TOÀN CỤC ---
ml_models = {}
STOPWORDS = set()

# --- CÁC HÀM HỖ TRỢ ---

def load_stopwords_global():
    global STOPWORDS
    try:
        if os.path.exists(STOPWORDS_FILE):
            with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        STOPWORDS.add(word)
                        STOPWORDS.add(word.replace(' ', '_'))
            print(f"Đã tải {len(STOPWORDS)} từ dừng.")
        else:
            print(f"Cảnh báo: Không tìm thấy file '{STOPWORDS_FILE}'.")
    except Exception as e:
        print(f"Lỗi khi đọc stopwords: {e}")

def text_preprocess(text):
    if not isinstance(text, str) or text is None:
        return ""
    
    # Chuẩn hóa Unicode & Regex làm sạch
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\S*@\S*\s?', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Chuyển thường & Bỏ khoảng trắng
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tách từ
    try:
        text = word_tokenize(text, format='text')
    except:
        pass
    
    # Loại bỏ Stopwords
    if text and STOPWORDS:
        words = text.split()
        clean_words = [w for w in words if w not in STOPWORDS]
        text = " ".join(clean_words)
    return text

def get_embeddings(text_list, max_len=256):
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
    
    # Sử dụng Mean Pooling
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# --- QUẢN LÝ LIFESPAN ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n--- SERVER STARTUP ---")
    
    load_stopwords_global()

    try:
        if not os.path.exists(MLP_MODEL_PATH) or not os.path.exists(LE_MODEL_PATH):
             print(f"LỖI: Không tìm thấy file model tại {MLP_MODEL_PATH}")
        else:
            ml_models['le'] = joblib.load(LE_MODEL_PATH)
            ml_models['mlp'] = joblib.load(MLP_MODEL_PATH)
            print("Đã tải MLP Model & Label Encoder.")
    except Exception as e:
        print(f"Lỗi tải file .pkl: {e}")

    try:
        print(f"Đang tải PhoBERT từ {MODEL_PATH}...")
        ml_models['tokenizer'] = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
        ml_models['bert'] = AutoModel.from_pretrained(MODEL_PATH)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ml_models['bert'].to(device)
        ml_models['device'] = device
        print(f"Đã tải xong PhoBERT (Device: {device})")
    except Exception as e:
        print(f"Lỗi tải PhoBERT: {e}")

    print("Server đã sẵn sàng!\n")
    yield
    ml_models.clear()

# --- APP & ROUTES ---
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(data: NewsInput):
    try:
        # 1. Tiền xử lý
        clean_title = text_preprocess(data.title)
        clean_content = text_preprocess(data.content) 

        # Trích xuất từ khóa
        full_text = clean_title + " " + clean_content
        words = full_text.split()
        most_common = Counter(words).most_common(10)
        keywords = [word.replace('_', ' ') for word, count in most_common]

        # 2. Tạo đặc trưng
        token_count = len(clean_content.split())
        title_emb = get_embeddings([clean_title], max_len=64)
        content_emb = get_embeddings([clean_content], max_len=256)
        token_feat = np.array([[token_count]])
        
        # 3. Gộp & Dự đoán
        full_features = np.hstack([title_emb, content_emb, token_feat])
        
        mlp = ml_models['mlp']
        le = ml_models['le']
        
        pred_idx = mlp.predict(full_features)[0]
        label = le.inverse_transform([pred_idx])[0]
        
        probs = mlp.predict_proba(full_features)[0]
        confidence = probs[pred_idx]
        
        return {
            "status": "success",
            "prediction": label,
            "confidence": f"{confidence*100:.2f}%",
            "keywords": keywords
        }

    except Exception as e:
        print(f"Lỗi dự đoán: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)