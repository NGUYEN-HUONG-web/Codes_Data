 # Báo cáo đồ án cuối kỳ

 **Tên đề tài:** Xây dựng Transformer cho bài toán phân loại cảm xúc văn bản

 - Họ tên sinh viên: .....................................................
 - MSSV: .....................................................
 - Lớp: .....................................................
 - Giảng viên: .....................................................
 - Ngày nộp: .....................................................
 - Phiên bản báo cáo: 1.0

 ---

 ## 1. Mở đầu và phát biểu bài toán

 Bài toán: phân loại cảm xúc cho câu tiếng Anh ngắn thành 3 lớp: `negative`, `neutral`, `positive`.

 Mục tiêu học thuật: tự cài đặt thành phần Self-Attention và Feed-Forward (FFN) của Transformer, hiểu cách hoạt động của attention, thực hiện thí nghiệm so sánh với baseline MLP, trực quan hóa attention và phân tích lỗi.

 Phạm vi cài đặt: thành phần tự cài gồm hàm `scaled_dot_product_attention`, lớp `SelfAttention`, `FeedForwardNetwork`, `TransformerEncoderBlock`. Được phép dùng thư viện cho `nn.Embedding`, `nn.LayerNorm`, optimizer, loss và visualizations.

 Dữ liệu: tập giả lập 600 mẫu đã được cung cấp, chia train/val/test = 420/90/90, cân bằng 3 nhãn.

 ## 2. Mô tả kiến trúc mô hình

 Kiến trúc tổng thể (một khối):

 - Input: token IDs (đã được padding/truncate tới `max_len` = 20)
 - Embedding layer: chuyển token ID → vector `d_model`
 - PositionalEncoding: cộng thông tin vị trí (sinusoidal)
 - 1 × TransformerEncoderBlock:
   - Self-Attention (Q,K,V)
   - Add & LayerNorm (residual)
   - Feed-Forward Network: Linear(d_model, d_ff) → ReLU → Linear(d_ff, d_model)
   - Add & LayerNorm
 - Classifier head: mean pooling → Linear(d_model, num_classes)

 Lý do chỉ dùng 1 khối encoder: do tập dữ liệu nhỏ (600 mẫu), nhiều block sẽ gây overfitting và khó phân tích attention sâu; mục tiêu là hiểu chi tiết 1 khối.

 Vai trò các thành phần: Embedding (mã hóa token), PositionalEncoding (thông tin vị trí), Self-Attention (tính tương quan giữa token), Add&LayerNorm (ổn định học), FFN (năng lực phi tuyến vị trí), Classifier (xuất logits).

 ## 3. Chi tiết cài đặt Self-Attention và FFN

 ### Scaled Dot-Product Attention

 Ta tính:

 $$scores = \frac{Q K^T}{\sqrt{d_k}}$$

 $$weights = \mathrm{softmax}(scores)$$

 $$output = weights \cdot V$$

 Shapes: Q,K,V: (B, L, d_k) → scores: (B, L, L) → output: (B, L, d_k).

 Trong code: hàm `scaled_dot_product_attention(Q,K,V)` trong `model.py` thực hiện các bước trên và trả về `(output, weights)`.

 ### Feed-Forward Network (FFN)

 Thiết kế: Linear(d_model, d_ff) → ReLU → Linear(d_ff, d_model).

 Trong code: lớp `FeedForwardNetwork` trong `model.py` với `fc1` và `fc2`.

 ### EncoderBlock

 Trong `TransformerEncoderBlock`, luồng tính:
 1. attn_out, attn_weights = SelfAttention(x)
 2. x = LayerNorm(x + attn_out)
 3. ffn_out = FFN(x)
 4. x = LayerNorm(x + ffn_out)

 ## 4. Thiết lập thực nghiệm

 - Lệnh chính (relative paths):

 ```powershell
 pip install -r requirements_used.txt
 python data_utils.py --max_len 20 --show_stats
 python model.py
 python train.py         # huấn luyện mặc định
 python train.py --run_all
 python visualize.py --model results/model_Transformer_d64_ff128.pt --sentence "this film is absolutely terrible"
 python viz_auto.py      # chọn tự động các ví dụ và lưu heatmap
 python viz_negation.py  # tạo heatmap ví dụ phủ định
 ```

 - Siêu tham số chính (đã dùng trong thí nghiệm):

 | Tham số | Giá trị |
 |---|---:|
 | `max_len` | 20 |
 | `batch_size` | 32 |
 | `d_model` (mặc định) | 64 |
 | `d_ff` (mặc định) | 128 |
 | `lr` | 1e-3 |
 | `num_epochs` | 20 |
 | `seed` | 42 |

 Tái lập kết quả: đã đặt `torch.manual_seed(42)` trong `train.py`; dùng đường dẫn tương đối; môi trường lưu trong `requirements_used.txt`.

 ## 5. Kết quả thực nghiệm và so sánh

 Kết quả (tóm tắt từ `results/summary.json`):

 | Mô hình | d_model | d_ff | Train Acc | Val Acc | Test Acc | Final Train Loss |
 |---|---:|---:|---:|---:|---:|---:|
 | Transformer_d64_ff128 | 64 | 128 | 0.7119 | 0.7222 | 0.6444 | 0.6645 |

 _Ghi chú_: trong workspace hiện chỉ có 1 cấu hình Transformer chạy sẵn; nếu cần, chạy `python train.py --run_all` để thu 3 cấu hình + baseline MLP.

 Learning curve (file): `results/learning_curve_Transformer_d64_ff128.png`

 Phân tích ngắn:

 - Model đạt val acc ≈ 0.72, test acc ≈ 0.64. Trên tập nhỏ, độ chênh giữa train/val cho thấy mức overfitting vừa phải.
 - Tăng `d_model` và `d_ff` thường làm tăng năng lực mô hình nhưng cũng dễ overfit trên dữ liệu chỉ 600 mẫu.

 ## 6. Phân tích Attention

 Đã tạo các heatmap và phân tích ngắn trong `results/attention_report.txt`.

 ### Heatmap 1 (ví dụ đúng)

 - File: `results/attention_correct_0.png`
 - Câu: "we discussed the movie in class at home"
 - Dự đoán: neutral (đúng)
 - Nhận xét: attention tập trung vào từ 'discussed' — mô hình chú ý từ này khi hình thành ngữ cảnh.

 ### Heatmap 2 (ví dụ sai)

 - File: `results/attention_incorrect_3.png`
 - Câu: "the film was remarkably fun now"
 - Dự đoán: negative (sai), nhãn thật: positive
 - Nhận xét: attention tập trung vào 'remarkably' (từ mang sắc thái) nhưng mô hình vẫn dự đoán sai — có thể do training không đủ hoặc từ vị trí/xuất hiện dữ liệu tương tự ít.

 ### Heatmap 3 (phủ định)

 - File: `results/attention_negation.png`
 - Câu: "i do not like this movie"
 - Dự đoán: negative
 - Nhận xét: top-attended token là 'do' theo báo cáo; cần kiểm tra xem attention có liên kết 'not' với 'like' hay không — nếu không, model chưa xử lý phủ định tốt.

 ## 7. Error Analysis

 Lấy 5 ví dụ mẫu model phân loại sai (nên chèn bảng 5–10 câu cụ thể). Ví dụ (mẫu):

 | STT | Câu | Nhãn đúng | Nhãn dự đoán | Ghi chú |
 |---:|---|---|---|---|
 | 1 | the film was remarkably fun now | positive | negative | model nhầm do dữ liệu tương tự ít |

 (Bạn nên chạy script nhỏ để thu 5–10 câu sai và dán vào bảng này.)

 Đề xuất cải thiện:

 - Sử dụng pretrained embeddings hoặc fine-tune từ BERT để cải thiện hiểu ngữ cảnh.
 - Augment dữ liệu hoặc thêm luật xử lý phủ định.

 ## 8. Kết luận

 - Đã tự cài đặt scaled dot-product attention và FFN, kiểm chứng bằng unit tests.
 - Mô hình Transformer đơn khối hoạt động hợp lý trên tập nhỏ nhưng giới hạn do dung lượng dữ liệu.
 - Visualization giúp xác định từ mô hình tập trung (hữu ích cho phân tích lỗi và cải tiến).

 ## 9. Tài liệu tham khảo

 - Vaswani et al., "Attention Is All You Need", 2017.
 - PyTorch documentation, torch.nn.

 ## Phụ lục

 - `results/summary.json` (tóm tắt kết quả)
 - `results/attention_report.txt` (ghi chú heatmap)
 - `requirements_used.txt` (các package hiện tại)

 ---

 *Ghi chú:* đây là bản draft tự động. Bạn nên kiểm tra lại các số liệu (nếu đã chạy thêm cấu hình), sửa phần Họ tên, MSSV, Lớp, Giảng viên, và bổ sung bảng Error Analysis với 5–10 câu sai thực tế.
