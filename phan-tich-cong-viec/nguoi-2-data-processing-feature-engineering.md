# PHÂN TÍCH CÔNG VIỆC NGƯỜI 2
# Vai trò: Data Processing & Feature Engineering Engineer

---

# 1. Mục tiêu phần việc

Người 2 chịu trách nhiệm chính cho phần **làm sạch dữ liệu, tiền xử lý, chia tập dữ liệu, augmentation, trích xuất đặc trưng, giảm chiều và trực quan hóa đặc trưng**.

Đây là phần nối giữa dữ liệu thô của Người 1 và mô hình học sâu của Người 3, Người 4.

Mục tiêu bắt buộc:

- Biến dữ liệu raw thành dữ liệu clean.
- Đảm bảo dataset sạch còn hơn 10.000 ảnh hợp lệ.
- Chuẩn hóa ảnh đầu vào cho MobileNetV2 và ResNet50.
- Chia dữ liệu thành Train/Validation/Test theo tỉ lệ phù hợp.
- Thiết kế augmentation hợp lý.
- Trích xuất đặc trưng ảnh.
- Giảm chiều bằng PCA và t-SNE hoặc UMAP.
- Trực quan hóa kết quả xử lý và đặc trưng.

Người 2 phải trả lời được các câu hỏi:

- Ảnh lỗi được loại như thế nào?
- Ảnh trùng được phát hiện như thế nào?
- Ảnh mờ được phát hiện như thế nào?
- Vì sao resize về 224x224?
- Vì sao cần normalize?
- Vì sao chỉ augmentation train set?
- Vì sao cần chia train/validation/test?
- PCA/t-SNE thể hiện điều gì?
- Feature vector được lấy từ đâu?

---

# 2. Vai trò của xử lý dữ liệu

Dữ liệu raw sau khi crawl thường không thể đưa trực tiếp vào mô hình.

Dữ liệu raw có thể chứa:

- Ảnh bị lỗi.
- Ảnh không đọc được.
- Ảnh quá nhỏ.
- Ảnh bị trùng.
- Ảnh mờ.
- Ảnh sai class.
- Ảnh có định dạng khác nhau.
- Ảnh có kích thước khác nhau.
- Ảnh có nền phức tạp.
- Class imbalance.

Nếu không xử lý tốt:

- Mô hình học sai.
- Training bị lỗi.
- Accuracy thấp.
- Test accuracy không đáng tin.
- Kết quả visualization không rõ ràng.

Người 2 phải đảm bảo dữ liệu đầu vào cho mô hình:

```text
Sạch -> Chuẩn hóa -> Chia đúng -> Có augmentation hợp lý -> Có đặc trưng trực quan hóa được
```

---

# 3. Kiến thức cần nắm

## 3.1 Data cleaning là gì?

Data cleaning là quá trình phát hiện và loại bỏ dữ liệu không phù hợp.

Trong project ảnh, cleaning gồm:

- Loại ảnh không đọc được.
- Loại ảnh sai định dạng.
- Loại ảnh quá nhỏ.
- Loại ảnh trùng.
- Loại ảnh quá mờ.
- Loại ảnh không liên quan đến class.

Data cleaning giúp:

- Tăng chất lượng dataset.
- Giảm nhiễu.
- Giảm overfitting.
- Giúp mô hình học đặc trưng đúng hơn.

## 3.2 Preprocessing là gì?

Preprocessing là quá trình biến ảnh về định dạng phù hợp với mô hình.

Với MobileNetV2 và ResNet50:

```text
Input shape: 224 x 224 x 3
```

Các bước thường gồm:

- Đọc ảnh.
- Convert về RGB.
- Resize về 224x224.
- Convert sang tensor.
- Normalize hoặc dùng `preprocess_input`.

## 3.3 Data augmentation là gì?

Data augmentation là kỹ thuật tạo biến thể ảnh từ ảnh gốc trong quá trình training.

Ví dụ:

- Xoay nhẹ.
- Zoom nhẹ.
- Dịch ảnh nhẹ.
- Thay đổi độ sáng.
- Thay đổi tương phản.

Mục tiêu:

- Tăng độ đa dạng dữ liệu.
- Giảm overfitting.
- Giúp mô hình bền hơn với ảnh thực tế.

Lưu ý:

```text
Không augmentation validation/test set.
```

Vì validation/test phải mô phỏng dữ liệu thật, dùng để đánh giá khách quan.

## 3.4 Train/Validation/Test là gì?

| Tập | Mục đích |
|---|---|
| Train | Dùng để học trọng số mô hình |
| Validation | Dùng để chọn hyperparameter và theo dõi overfitting |
| Test | Dùng để đánh giá cuối cùng |

Tỉ lệ đề xuất:

```text
Train: 70%
Validation: 15%
Test: 15%
```

Yêu cầu:

- Split theo kiểu stratified.
- Mỗi class giữ tỉ lệ tương đối giống nhau trong cả 3 tập.
- Không để ảnh trùng xuất hiện ở nhiều tập.

## 3.5 Feature extraction là gì?

Feature extraction là quá trình biến ảnh thành vector đặc trưng.

Có hai loại đặc trưng trong project:

### Đặc trưng thủ công/mức ảnh

Ví dụ:

- Kích thước ảnh.
- Phân bố màu.
- Độ sáng.
- Độ tương phản.
- Độ sắc nét.

Các đặc trưng này giúp mô tả dữ liệu và giải thích bước cleaning.

### Đặc trưng học sâu

Lấy từ CNN backbone:

- MobileNetV2 without top.
- ResNet50 without top.
- GlobalAveragePooling.

Output là embedding vector đại diện cho ảnh.

## 3.6 PCA là gì?

PCA là phương pháp giảm chiều tuyến tính.

Mục tiêu:

- Biến vector nhiều chiều thành 2D hoặc 3D để vẽ.
- Giữ lại hướng có phương sai lớn nhất.
- Quan sát dữ liệu có xu hướng phân cụm theo class hay không.

PCA phù hợp để:

- Trực quan hóa nhanh.
- Giải thích phương sai dữ liệu.
- Phát hiện outlier.

## 3.7 t-SNE là gì?

t-SNE là phương pháp giảm chiều phi tuyến.

Mục tiêu:

- Bảo toàn quan hệ gần nhau giữa các điểm dữ liệu.
- Hiển thị cụm class rõ hơn PCA trong nhiều trường hợp.

Lưu ý:

- t-SNE chủ yếu dùng để trực quan hóa.
- Không nên dùng t-SNE để kết luận định lượng tuyệt đối.
- Kết quả có thể thay đổi theo tham số và random seed.

## 3.8 UMAP là gì?

UMAP cũng là phương pháp giảm chiều phi tuyến.

So với t-SNE:

- Chạy nhanh hơn trên dữ liệu lớn.
- Có thể giữ cấu trúc tổng thể tốt hơn.
- Phù hợp để visualization embedding.

Nếu không dùng UMAP thì dùng t-SNE là đủ.

---

# 4. Quy trình làm sạch dữ liệu

## 4.1 Input từ Người 1

Người 2 nhận:

```text
data/raw/
data/metadata/images_metadata.csv
data/metadata/classes.csv
```

## 4.2 Output cần tạo

Người 2 tạo:

```text
data/clean/
data/split/train/
data/split/val/
data/split/test/
outputs/figures/
outputs/metrics/
```

## 4.3 Kiểm tra ảnh lỗi

Ảnh lỗi là ảnh không thể đọc bằng PIL/OpenCV.

Nguyên nhân:

- Download chưa xong.
- File không phải ảnh.
- File bị hỏng.
- URL trả về HTML thay vì image.

Cách kiểm tra:

- Dùng PIL `Image.verify()`.
- Dùng OpenCV `cv2.imread()`.
- Nếu không đọc được thì đánh dấu corrupted.

Metadata cập nhật:

```text
is_corrupted = true
final_status = removed
```

## 4.4 Kiểm tra kích thước ảnh

Ảnh quá nhỏ thường không đủ thông tin.

Ngưỡng đề xuất:

```text
width >= 64
height >= 64
```

Nếu ảnh nhỏ hơn:

```text
final_status = removed
remove_reason = too_small
```

## 4.5 Kiểm tra ảnh trùng

Phương pháp đề xuất:

```text
perceptual hashing
```

Các loại hash có thể dùng:

- average hash.
- difference hash.
- perceptual hash.
- wavelet hash.

Ý tưởng:

- Chuyển ảnh thành fingerprint.
- Ảnh giống nhau sẽ có hash gần giống nhau.
- Nếu khoảng cách hash nhỏ hơn ngưỡng thì xem là trùng.

Lý do không dùng tên file:

- Cùng ảnh có thể có tên khác.
- Cùng ảnh có thể được crawl từ nhiều nguồn.
- Cùng ảnh có thể resize hoặc nén lại.

## 4.6 Kiểm tra ảnh mờ

Phương pháp phổ biến:

```text
Variance of Laplacian
```

Ý tưởng:

- Ảnh rõ có nhiều cạnh sắc nét.
- Ảnh mờ có ít thay đổi biên.
- Laplacian đo mức thay đổi cạnh.
- Variance thấp nghĩa là ảnh có thể bị mờ.

Quy trình:

```text
Ảnh RGB -> grayscale -> Laplacian -> variance -> so sánh threshold
```

Lưu ý:

- Threshold cần thử nghiệm.
- Không nên loại quá mạnh vì ảnh thực tế có thể hơi mờ.
- Nên lưu một số sample ảnh bị loại để minh họa.

## 4.7 Kiểm tra class imbalance

Sau cleaning, thống kê lại số ảnh mỗi class.

Nếu class nào quá ít:

- Báo Người 1 crawl thêm.
- Hoặc điều chỉnh class nếu thật sự không đủ dữ liệu.

Không nên để:

```text
Class lớn nhất / class nhỏ nhất > 3
```

Nếu lệch quá nhiều, cần xử lý.

---

# 5. Preprocessing pipeline

## 5.1 Resize

MobileNetV2 và ResNet50 thường dùng input:

```text
224 x 224 x 3
```

Vì vậy cần resize toàn bộ ảnh về:

```text
224 x 224
```

Cần lưu ý:

- Resize có thể làm méo ảnh nếu không giữ tỉ lệ.
- Có thể dùng center crop hoặc padding nếu muốn giữ tỉ lệ.
- Trong project 3 tuần, resize trực tiếp 224x224 là chấp nhận được nếu kết quả tốt.

## 5.2 Convert RGB

Ảnh crawl có thể là:

- RGB.
- RGBA.
- Grayscale.
- CMYK.

Mô hình pretrained thường cần 3 channel RGB.

Do đó:

```text
Convert tất cả ảnh về RGB
```

## 5.3 Normalize

Normalize giúp đưa pixel về miền giá trị phù hợp.

Cách 1:

```text
pixel / 255.0
```

Cách 2:

```text
preprocess_input() theo từng backbone
```

Với transfer learning, nên dùng `preprocess_input()` tương ứng:

- MobileNetV2 preprocess.
- ResNet50 preprocess.

Vì các mô hình pretrained được train với preprocessing riêng.

---

# 6. Augmentation pipeline

## 6.1 Kỹ thuật đề xuất

| Kỹ thuật | Giá trị đề xuất | Ý nghĩa |
|---|---|---|
| Rotation | 10-15 độ | Mô phỏng góc chụp lệch nhẹ |
| Zoom | 0.1-0.15 | Mô phỏng khoảng cách gần/xa |
| Width shift | 0.1 | Mô phỏng lệch ngang |
| Height shift | 0.1 | Mô phỏng lệch dọc |
| Brightness | 0.8-1.2 | Mô phỏng ánh sáng khác nhau |
| Contrast | nhẹ | Mô phỏng điều kiện môi trường |

## 6.2 Kỹ thuật nên tránh

Không nên dùng quá mạnh:

- Flip ngang/dọc bừa bãi.
- Rotation 90 độ.
- Crop mất phần biển báo.
- Color jitter quá mạnh.

Lý do:

- Biển báo có hướng và ký hiệu cụ thể.
- Flip có thể làm biển rẽ trái thành rẽ phải.
- Xoay quá nhiều làm ảnh không còn thực tế.

## 6.3 Augmentation experiment

Người 2 nên phối hợp Người 3/4 để so sánh:

| Cấu hình | Validation accuracy |
|---|---:|
| No augmentation | |
| Rotation + zoom | |
| Brightness + contrast | |
| Full augmentation | |

Mục tiêu:

- Chọn augmentation giúp model tốt hơn.
- Tránh augmentation gây giảm accuracy.

---

# 7. Chia Train/Validation/Test

## 7.1 Tỉ lệ

Tỉ lệ đề xuất:

```text
Train: 70%
Validation: 15%
Test: 15%
```

Ví dụ dataset sạch có 10.000 ảnh:

| Tập | Số ảnh |
|---|---:|
| Train | 7.000 |
| Validation | 1.500 |
| Test | 1.500 |

## 7.2 Stratified split

Stratified split giúp mỗi tập có tỉ lệ class tương tự dataset gốc.

Ví dụ class `stop` chiếm 10% dataset:

- Train cũng khoảng 10% stop.
- Validation cũng khoảng 10% stop.
- Test cũng khoảng 10% stop.

## 7.3 Tránh data leakage

Data leakage là khi thông tin từ test set bị lọt vào training.

Ví dụ:

- Ảnh trùng xuất hiện cả train và test.
- Augmented version của ảnh train xuất hiện trong test.
- Dùng test set để chọn hyperparameter.

Người 2 cần đảm bảo:

```text
Split sau khi loại duplicate.
Augmentation chỉ áp dụng trong training.
Test set chỉ dùng để đánh giá cuối.
```

---

# 8. Trích xuất đặc trưng

## 8.1 Đặc trưng mức ảnh

Các đặc trưng nên tính:

| Đặc trưng | Ý nghĩa |
|---|---|
| width | Chiều rộng ảnh |
| height | Chiều cao ảnh |
| aspect_ratio | Tỉ lệ rộng/cao |
| mean_brightness | Độ sáng trung bình |
| contrast | Độ tương phản |
| sharpness | Độ sắc nét |
| mean_R/G/B | Trung bình màu |

Các đặc trưng này dùng để:

- Thống kê dữ liệu.
- Phát hiện ảnh bất thường.
- Giải thích chất lượng dữ liệu.

## 8.2 Đặc trưng từ CNN

Feature vector có thể lấy từ:

```text
MobileNetV2 without top
ResNet50 without top
```

Quy trình:

```text
Ảnh 224x224 -> Backbone CNN -> GlobalAveragePooling -> Feature vector
```

Feature vector thể hiện đặc trưng cấp cao:

- Hình dạng biển báo.
- Màu sắc.
- Ký hiệu.
- Cấu trúc cạnh.
- Vùng quan trọng trong ảnh.

## 8.3 Feature trước và sau fine-tuning

Nếu có thời gian, nên vẽ:

- Feature từ model pretrained chưa fine-tune.
- Feature từ model sau fine-tune.

Kỳ vọng:

```text
Sau fine-tuning, các điểm cùng class gom cụm rõ hơn.
```

---

# 9. Trực quan hóa cần có

Người 2 cần tạo các biểu đồ:

| Biểu đồ | Mục đích |
|---|---|
| Clean vs removed images | Cho thấy quá trình cleaning |
| Class distribution after cleaning | Kiểm tra cân bằng class |
| Image resolution distribution | Mô tả kích thước ảnh |
| Sample removed images | Minh họa ảnh bị loại |
| Sample augmented images | Minh họa augmentation |
| PCA 2D plot | Trực quan hóa feature |
| t-SNE/UMAP 2D plot | Trực quan hóa cụm class |
| PCA explained variance | Giải thích lượng thông tin giữ lại |

---

# 10. Deliverables bắt buộc

| STT | Deliverable | Mô tả |
|---|---|---|
| 1 | `src/data/clean_images.py` | Script lọc ảnh lỗi/mờ/quá nhỏ |
| 2 | `src/data/remove_duplicates.py` | Script loại ảnh trùng |
| 3 | `src/data/split_dataset.py` | Script chia train/val/test |
| 4 | `src/features/preprocessing.py` | Pipeline preprocessing |
| 5 | `src/features/augmentation.py` | Pipeline augmentation |
| 6 | `src/features/extract_features.py` | Script trích xuất feature |
| 7 | `src/features/visualize_features.py` | Script vẽ PCA/t-SNE |
| 8 | `data/clean/` | Dataset sạch |
| 9 | `data/split/` | Dataset đã chia |
| 10 | `outputs/figures/feature_pca.png` | PCA plot |
| 11 | `outputs/figures/feature_tsne.png` | t-SNE plot |
| 12 | `outputs/metrics/cleaning_summary.csv` | Thống kê cleaning |

---

# 11. Tiêu chí hoàn thành

Người 2 hoàn thành khi:

- Dataset sạch còn hơn 10.000 ảnh.
- Không còn ảnh lỗi trong dataset sạch.
- Đã loại duplicate ở mức hợp lý.
- Đã chia train/validation/test theo tỉ lệ 70/15/15.
- Split là stratified.
- Augmentation chỉ dùng cho train set.
- Có preprocessing pipeline dùng được cho cả MobileNetV2 và ResNet50.
- Có PCA/t-SNE hoặc UMAP visualization.
- Có biểu đồ trước/sau cleaning và augmentation.

---

# 12. Rủi ro và cách xử lý

| Rủi ro | Nguyên nhân | Cách xử lý |
|---|---|---|
| Sau clean còn dưới 10.000 ảnh | Loại quá nhiều ảnh | Báo Người 1 crawl thêm |
| Class imbalance | Một số class bị loại nhiều | Crawl bổ sung class thiếu, dùng augmentation hợp lý |
| Loại ảnh mờ quá mạnh | Threshold blur quá cao | Hạ threshold, kiểm tra sample |
| Ảnh trùng lọt qua | Hash threshold quá lỏng | Điều chỉnh perceptual hash distance |
| Data leakage | Split trước khi remove duplicate | Remove duplicate trước split |
| Augmentation làm sai nhãn | Flip/xoay quá mạnh | Giảm augmentation, tránh flip class định hướng |

---

# 13. Kiến thức bảo vệ cần nắm

## 13.1 Vì sao phải làm sạch dữ liệu?

Vì dữ liệu crawl từ Internet có nhiều ảnh lỗi, ảnh trùng, ảnh mờ và ảnh sai nội dung. Làm sạch giúp mô hình học từ dữ liệu chất lượng hơn, giảm nhiễu và tăng độ tin cậy của kết quả đánh giá.

## 13.2 Vì sao resize về 224x224?

Vì MobileNetV2 và ResNet50 pretrained thường nhận input ảnh 224x224x3. Resize giúp toàn bộ ảnh có cùng kích thước để đưa vào mô hình.

## 13.3 Vì sao dùng augmentation?

Augmentation tạo thêm biến thể ảnh trong quá trình training, giúp mô hình giảm overfitting và bền hơn với các điều kiện thực tế như góc chụp, ánh sáng và khoảng cách khác nhau.

## 13.4 Vì sao không augmentation test set?

Test set dùng để đánh giá khách quan mô hình trên dữ liệu chưa thấy. Nếu augmentation test set hoặc dùng test set trong training thì kết quả đánh giá không còn đáng tin.

## 13.5 PCA/t-SNE dùng để làm gì?

PCA và t-SNE giúp giảm chiều feature vector xuống 2D để trực quan hóa. Qua đó nhóm có thể quan sát các ảnh cùng class có xu hướng gom cụm hay không và các class nào dễ bị chồng lấn.

---

# 14. Câu hỏi bảo vệ mẫu

## Câu 1: Nhóm em làm sạch dữ liệu như thế nào?

Trả lời gợi ý:

```text
Nhóm em kiểm tra ảnh không đọc được, ảnh quá nhỏ, ảnh trùng lặp bằng perceptual hashing và ảnh mờ bằng variance of Laplacian. Các ảnh không đạt tiêu chí được đánh dấu trong metadata và loại khỏi dataset sạch.
```

## Câu 2: Vì sao cần chia Train/Validation/Test?

Trả lời gợi ý:

```text
Train set dùng để huấn luyện mô hình, validation set dùng để hiệu chỉnh hyperparameter và theo dõi overfitting, test set dùng để đánh giá cuối cùng. Việc tách riêng giúp kết quả đánh giá khách quan hơn.
```

## Câu 3: Làm sao tránh data leakage?

Trả lời gợi ý:

```text
Nhóm em loại ảnh trùng trước khi chia dữ liệu, dùng stratified split và chỉ áp dụng augmentation cho train set. Test set không được dùng để chọn hyperparameter hay huấn luyện.
```

## Câu 4: Vì sao dùng PCA và t-SNE?

Trả lời gợi ý:

```text
Feature vector từ CNN có số chiều cao nên khó quan sát trực tiếp. PCA và t-SNE giúp giảm xuống 2 chiều để trực quan hóa feature space, từ đó nhận xét các class có tách biệt tốt hay bị chồng lấn.
```

## Câu 5: Augmentation có thể làm giảm accuracy không?

Trả lời gợi ý:

```text
Có. Nếu augmentation quá mạnh hoặc làm thay đổi ý nghĩa biển báo, mô hình có thể học sai. Vì vậy nhóm chỉ dùng augmentation nhẹ như xoay nhỏ, zoom nhẹ, thay đổi sáng tương phản vừa phải và tránh flip với các class có hướng.
```

---

# 15. Checklist cá nhân Người 2

- [ ] Đã nhận dataset raw và metadata từ Người 1.
- [ ] Đã loại ảnh lỗi.
- [ ] Đã loại ảnh quá nhỏ.
- [ ] Đã xử lý ảnh trùng.
- [ ] Đã kiểm tra ảnh mờ.
- [ ] Dataset clean còn hơn 10.000 ảnh.
- [ ] Đã chia train/validation/test.
- [ ] Split theo stratified.
- [ ] Đã tạo preprocessing pipeline.
- [ ] Đã tạo augmentation pipeline.
- [ ] Đã trích xuất feature vector.
- [ ] Đã vẽ PCA.
- [ ] Đã vẽ t-SNE hoặc UMAP.
- [ ] Đã bàn giao dataset split cho Người 3 và Người 4.
- [ ] Đã chuẩn bị giải thích khi bảo vệ.

---

# 16. Kết luận phần Người 2

Người 2 đảm bảo dữ liệu của project đạt chất lượng đủ tốt để huấn luyện mô hình.

Khi bảo vệ, Người 2 cần nhấn mạnh:

- Dữ liệu raw đã được làm sạch có hệ thống.
- Dataset clean còn hơn 10.000 ảnh hợp lệ.
- Train/Validation/Test được chia đúng và tránh data leakage.
- Augmentation được dùng hợp lý.
- Đặc trưng được trích xuất và trực quan hóa bằng PCA/t-SNE.
- Kết quả feature visualization hỗ trợ giải thích khả năng phân loại của mô hình.
