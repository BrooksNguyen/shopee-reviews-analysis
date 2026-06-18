# Phân Tích Cảm Xúc Đánh Giá Của Người Dùng Trên Nền Tảng Thương Mại Điện Tử

## 1. Giới thiệu bài toán (Problem Formulation)
Trong kỷ nguyên thương mại điện tử, đánh giá của người dùng (user reviews) đóng vai trò sống còn trong quyết định mua hàng. Tuy nhiên, việc đọc thủ công hàng ngàn đánh giá là không khả thi. 
Bài toán đặt ra: **Phân tích và phân loại tự động cảm xúc (Sentiment Analysis)** của các bình luận bằng tiếng Việt từ Shopee thành hai cực: **Tích cực (Positive)** và **Tiêu cực (Negative)**. 

Dự án này sử dụng 9,599 bình luận thực tế, được dán nhãn sẵn từ tệp `shopee_reviews_dataset.jsonl` (nguồn Kaggle). Nhãn `positive` được ánh xạ thành `1` và `negative` được ánh xạ thành `0`.

---

## 2. Pipeline Xử lý (Processing Pipeline)
Hệ thống được thiết kế với một đường ống (pipeline) rõ ràng:

1. **Thu thập và Chia dữ liệu:** Dữ liệu được đọc từ JSONL và chia theo tỷ lệ 80% (Train), 10% (Validation), 10% (Test) với Stratified sampling để đảm bảo tỷ lệ nhãn cân bằng.
2. **Tiền xử lý văn bản (Preprocessing):** (Thực hiện trong `utils.py`)
   - **Làm sạch (Cleaning):** Loại bỏ HTML tags, ký tự đặc biệt, dấu câu dư thừa.
   - **Chuẩn hóa từ kéo dài (Elongated Words):** Nhận diện và loại bỏ các ký tự lặp lại vô nghĩa (ví dụ: `ngonnnn` -> `ngon`, `quáaaa` -> `quá`) để quy về từ gốc chuẩn.
   - **Chuẩn hóa Teencode:** Chuyển đổi các từ viết tắt phổ biến của Gen Z (ví dụ: `ko`, `sp`, `đẹp vcl`) thành ngôn ngữ chuẩn bằng một từ điển định nghĩa sẵn.
   - **Tách từ (Word Segmentation):** Sử dụng `underthesea` để nối các âm tiết thành từ ghép (ví dụ: `sinh_viên` thay vì `sinh`, `viên`).
   - **Phân tích & Lọc độ dài:** Thống kê độ dài câu (word count) của toàn bộ dữ liệu trước và sau tiền xử lý; tự động loại bỏ các câu quá ngắn (dưới 2 từ) không mang nội dung cảm xúc thực tế nhằm giảm thiểu nhiễu cho mô hình.
3. **Trích xuất đặc trưng (Feature Extraction):**
   - *Baseline:* Sử dụng `TfidfVectorizer` để chuyển đổi văn bản thành ma trận tần suất dựa trên N-grams.
   - *Deep Learning:* Sử dụng `Word Embedding` layer của PyTorch và kiến trúc LSTM.
   - *Transformer:* Sử dụng `AutoTokenizer` của HuggingFace (PhoBERT) để token hóa theo subword.
4. **Huấn luyện mô hình (Modeling):** Huấn luyện đồng thời các mô hình để so sánh: Naive Bayes, Logistic Regression, Bi-LSTM, và PhoBERT.

---

## 3. Kết quả và So sánh Mô hình (Results & Comparison)

Dưới đây là bảng đánh giá hiệu suất của các mô hình trên tập Test:

| Mô hình | Accuracy (%) | F1-Score (%) | Thời gian Train | Phân tích |
|---|---|---|---|---|
| **Naive Bayes** | ~93.5% | ~93.1% | Rất nhanh (< 0.1s) | Hoạt động cực kỳ tốt nhờ khả năng bắt các từ khóa mang tính phân cực độc lập ("tuyệt_vời", "tệ"). |
| **Logistic Regression** | ~93.0% | ~92.5% | Nhanh (~0.1s) | Cùng mức độ với NB, phạt các nhiễu tốt nhờ cơ chế Regularization. |
| **PyTorch Bi-LSTM** | ~91.6% | ~91.1% | Chậm (~2 phút) | Học được ngữ cảnh và thứ tự từ (sequence), dễ bị Overfit trên lượng data nhỏ. |
| **PhoBERT (10% Fine-tuned)** | ~92.0%* | ~91.5%* | Rất chậm (CPU) | Bắt được ngữ cảnh tiếng Việt tốt nhất, nhưng do giới hạn phần cứng nên chỉ train demo trên 10% dữ liệu. |
| **DistilBERT (10% Fine-tuned)** | ~90.5%* | ~90.0%* | Trung bình | Kiến trúc rút gọn của BERT, tốc độ huấn luyện nhanh hơn PhoBERT trên CPU nhưng độ chính xác tiếng Việt kém hơn một chút do là mô hình đa ngôn ngữ tổng quát. |
| **PhoBERT (Pre-trained Full)** | ~94.5% | ~94.0% | **0 giây** (Chỉ inference) | Sử dụng trọng số đã được huấn luyện sẵn trên 30k reviews (`wonrax/phobert-base-vietnamese-sentiment`). Đạt hiệu năng cao nhất nhờ học được toàn bộ dữ liệu khổng lồ ngoài miền mà không cần tài nguyên huấn luyện local. |

**Nhận xét chuyên sâu:** 
1. **Đánh giá về thời gian và độ phức tạp:** Fine-tune các mô hình Transformer lớn từ đầu trên CPU là không khả thi cho môi trường làm bài thực hành nhanh. Việc lựa chọn **DistilBERT** làm giải pháp thay thế giúp đẩy nhanh tốc độ chạy, trong khi việc dùng trực tiếp mô hình **Pre-trained Sentiment PhoBERT** chứng minh hiệu quả tuyệt vời của phương pháp Transfer Learning khi không có GPU để train.
2. **Tại sao Baseline (ML truyền thống) vẫn rất mạnh?** Trong ngữ cảnh đánh giá Shopee, các câu nhận xét thường rất ngắn gọn và trực diện (ví dụ: "Sản phẩm tốt", "Giao hàng quá chậm"). Điều này giải thích tại sao các phương pháp Baseline dựa trên tần suất từ (TF-IDF) lại có hiệu suất rất cao và khó bị đánh bại bởi các mô hình Deep Learning tự huấn luyện trên tập dữ liệu nhỏ.

---

## 4. Phân tích lỗi (Error Analysis)
Mặc dù Accuracy cao (> 93%), mô hình vẫn mắc phải một số lỗi sai (False Positives/False Negatives) do:
1. **Câu châm biếm (Sarcasm):** Ví dụ: *"Giao hàng siêu nhanh luôn, đặt từ mùng 1 mà mùng 10 đã có hàng"*. Các từ "siêu nhanh" khiến TF-IDF dự đoán đây là Tích cực, trong khi ngữ nghĩa là Tiêu cực.
2. **Lỗi chính tả và Teencode mới:** Những từ lóng mới xuất hiện mà từ điển trong `utils.py` chưa cập nhật sẽ trở thành `<UNK>` (Unknown word) đối với mô hình LSTM/PhoBERT, làm mất đặc trưng quan trọng.
3. **Đánh giá mâu thuẫn:** Người dùng cho 5 sao nhưng bình luận *"Chưa dùng nên chưa biết, cho 5 sao ủng hộ shop"*. Nhãn gốc bị nhiễu do hành vi người dùng.

---

## 5. Hướng phát triển (Future Work)
- **Cải tiến từ điển Teencode:** Thay vì hard-code, có thể sử dụng các bộ từ điển động từ API hoặc huấn luyện một mô hình Spell Correction chuyên biệt cho tiếng Việt.
- **Xử lý ngữ nghĩa phức tạp:** Sử dụng các mô hình LLM lớn hơn (như ChatGPT API) hoặc fine-tune toàn bộ tập dữ liệu (100%) với PhoBERT trên GPU để giải quyết bài toán mỉa mai, châm biếm.
- **Phân loại đa nhãn (Multi-aspect Sentiment):** Thay vì chỉ đánh giá Tích cực/Tiêu cực chung chung, có thể phân tích thành các khía cạnh: "Giao hàng", "Chất lượng sản phẩm", "Chăm sóc khách hàng".

---

## 6. Hướng Dẫn Chạy Dự Án

### Yêu cầu hệ thống:
```bash
pip install -r requirements.txt
```

### Chạy mã nguồn:
Chạy file `main.py` để huấn luyện 4 mô hình và mở giao diện kiểm thử trực tiếp trên terminal:
```bash
python3 main.py
```
*(Lưu ý: Mô hình PhoBERT sẽ mất khá nhiều thời gian nếu máy bạn không có GPU CUDA).*
