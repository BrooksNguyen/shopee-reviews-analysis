import re
from underthesea import word_tokenize

# Tu dien teencode de chuan hoa review
TEENCODE_DICT = {
    "ko": "không",
    "k": "không",
    "khg": "không",
    "khong": "không",
    "k0": "không",
    "kg": "không",
    "sp": "sản phẩm",
    "sapham": "sản phẩm",
    "sanpham": "sản phẩm",
    "đc": "được",
    "dc": "được",
    "đuoc": "được",
    "chuan": "chuẩn",
    "iu": "yêu",
    "nv": "nhân viên",
    "tks": "cảm ơn",
    "thanks": "cảm ơn",
    "tk": "cảm ơn",
    "camon": "cảm ơn",
    "ok": "tốt",
    "oke": "tốt",
    "oki": "tốt",
    "rep": "trả lời",
    "fb": "phản hồi",
    "sd": "sử dụng",
    "qa": "quá",
    "dep": "đẹp",
    "shop": "cửa hàng",
    "sop": "cửa hàng",
    "shp": "cửa hàng",
    "vs": "với",
    "m": "mình",
    "mik": "mình",
    "ac": "anh chị",
    "trc": "trước",
    "chua": "chưa",
    "đt": "điện thoại",
    "dt": "điện thoại",
    "ib": "nhắn tin",
    "bh": "bảo hành",
    "r": "rồi",
    "sz": "cỡ",
    "tl": "trả lời",
    "nhg": "nhưng",
    "thik": "thích",
    "thic": "thích",
    "wa": "quá",
    "gud": "tốt",
    "good": "tốt",
    "like": "thích",
    "cute": "dễ thương"
}

def clean_text(text):
    # Chuyen ve chu thuong
    text = str(text).lower().strip()
    
    # Xoa html tag neu co
    text = re.sub(r'<[^>]*>', ' ', text)
    
    # Chuan hoa tu keo dai (elongated words: ngonnnn -> ngon, đẹppppp -> đẹp)
    text = re.sub(r'(\w)\1{2,}', r'\1', text)
    
    # Xoa cac ky tu dac biet va dau cau
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Xoa khoang trang du thua
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def replace_teencode(text):
    # Tach tu tam thoi bang khoang trang
    words = text.split()
    # Chuyen doi teencode tu dict
    cleaned_words = [TEENCODE_DICT.get(w, w) for w in words]
    return " ".join(cleaned_words)

def preprocess_review(text):
    # Ham tong hop de tien xu ly
    text = clean_text(text)
    text = replace_teencode(text)
    try:
        # Tach tu tieng Viet bang underthesea (dung dau gach duoi lam dung chung)
        text = word_tokenize(text, format="text")
    except Exception:
        # Neu underthesea loi thi dung split thong thuong
        pass
    return text
