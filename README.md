# Phan Tich Cam Xuc Review Thuong Mai Dien Tu (Sentiment Analysis on E-commerce Reviews)

Day la project ca nhan phan tich va phan loai cam xuc cac danh gia cua nguoi dung tren nen tang thuong mai dien tu bang tieng Viet (Shopee, Tiki, Lazada, v.v.). Project nay so sanh hai phuong phap:
1. **Traditional Machine Learning (Baseline):** TF-IDF + Naive Bayes va Logistic Regression.
2. **Deep Learning (Advanced):** PyTorch Bi-LSTM ket hop Word Embedding.

---

## Cac Buoc Cai Dat & Chay Du An

### 1. Cai dat moi truong va thu vien thuoc yeu cau
Mo terminal tai thu muc du an va chay lenh de cai cac thu vien can thiet:
```bash
pip install -r requirements.txt
```

### 2. Du lieu su dung
Du an su dung bo du lieu **shopee_reviews_dataset.jsonl** duoc tai ve truc tiep tu Kaggle. Bo du lieu nay co 9.599 dong danh gia thuc te tu Shopee.vn, bao gom:
- **Cac cot du lieu:** `id`, `review` (noi dung comment), `rating` (so sao), va `label` (nhan cam xuc).
- **Chuyen doi nhan (Label Mapping):** 
  - Nhan `positive` duoc map ve gia tri `1` (Tich cuc).
  - Nhan `negative` duoc map ve gia tri `0` (Tieu cuc).
- **Chia tap du lieu:** Du lieu duoc chia theo ti le **80% Train / 10% Validation / 10% Test** phu hop cho viec huan luyen va danh gia khach quan.

### 3. Tien xu ly du lieu
- Chuyen ve chu thuong, loai bo tag HTML va ky tu dac biet.
- Chuan hoa teencode tieng Viet bang tu dien (`utils.py`).
- Tach tu tieng Viet bang thu vien `underthesea`.

### 4. Chay notebook hoac file script
- **Chay Notebook:** Mo notebook `sentiment_analysis.ipynb` bang Jupyter Notebook hoac VS Code.
- **Chay Script truc tiep:** Chay file `main.py` de train va chay chuong trinh tester tuong tac ngay tren terminal:
```bash
python3 main.py
```

---

## Cau Truc Du An

- `requirements.txt`: Danh sach thu vien can thiet.
- `utils.py`: Bo cong cu tien xu ly van ban, loc teencode, tach tu tieng Viet.
- `shopee_reviews_dataset.jsonl`: File du lieu goc tu Shopee.
- `main.py`: File chay chuong trinh train va test tuong tac tu terminal.
- `sentiment_analysis.ipynb`: Notebook huan luyen chi tiet kem truc quan hoa va bieu do.
- `README.md`: Huong dan nay.

---

## Ket Qua So Sanh Chinh

| Mo hinh | Accuracy (%) | F1-Score (%) | Thoi gian train (s) |
|---|---|---|---|
| Naive Bayes (Baseline) | 93.54% | 93.12% | 0.0025 |
| Logistic Regression (Baseline) | 93.02% | 92.50% | 0.0475 |
| PyTorch Bi-LSTM (Advanced) | 91.67% | 91.11% | 103.81 |

- **Nhan xet va phan tich:**
  - **Machine Learning truyen thong (Naive Bayes & Logistic Regression):** Dat do chinh xac rat cao (> 93%) va chay trong tich tac (duoi 0.05s) vi cac review e-commerce co cac tu khoa phan cuc cuc ky manh.
  - **PyTorch Bi-LSTM:** Dat do chinh xac ~91.67%, huan luyen lau hon nhung co kha nang bat ngu canh tot hon voi cac cau dai, cac cum tu phủ dinh dung truoc tinh tu.

- **Han che (Limitations):**
  - Bo tu dien Teencode hien tai duoc khai bao cung (hard-coded) trong `utils.py`, dan toi kho bao phu cac tu long, teencode moi phat sinh cua Gen Z sau nay.
  - Mo hinh chua nhan biet tot cac cau mang tinh cham biem, mia mai.

- **Dinh huong cai thien:**
  - Thay the Embedding tu train bang cac mo hinh tieng Viet da duoc pre-train quy mo lon (vi du nhu PhoBERT, ViBERT) va thuc hien tinh chinh (fine-tune) de nang cao do chinh xac ngu canh.
  - Tich hop them tu dien teencode tu dong cap nhat tu mang xa hoi de mo rong kha nang tien xu ly.
