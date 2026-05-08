# PHÂN CÔNG CÔNG VIỆC NHÓM 4 NGƯỜI
# Đề tài: Nhận diện biển báo giao thông bằng mô hình học sâu

---

# 1. Mục tiêu project

Project thực hiện bài toán:

```text
Traffic Sign Classification
```

Mục tiêu chính:

- Tự thu thập dữ liệu ảnh biển báo giao thông bằng crawling.
- Xây dựng bộ dữ liệu có số lượng lớn hơn 10.000 mẫu ảnh hợp lệ.
- Làm sạch, chuẩn hóa, thống kê và trực quan hóa dữ liệu.
- Trích xuất và trực quan hóa đặc trưng ảnh.
- Huấn luyện, hiệu chỉnh và đánh giá ít nhất 2 mô hình học máy/học sâu.
- Đạt độ chính xác trên tập test không nhỏ hơn 85%.
- So sánh hiệu quả mô hình bằng các chỉ số định lượng và biểu đồ.

Các mô hình chính được chọn:

- MobileNetV2
- ResNet50

Lý do chọn:

- Cả hai đều phù hợp với bài toán phân loại ảnh.
- Có thể sử dụng transfer learning từ ImageNet để tăng hiệu quả trong thời gian ngắn.
- MobileNetV2 nhẹ, nhanh, phù hợp triển khai thực tế.
- ResNet50 mạnh hơn, dùng để so sánh trade-off giữa accuracy và tốc độ.

---

# 2. Phạm vi công việc kỹ thuật

Project gồm 3 phần kỹ thuật bắt buộc:

## 2.1 Thu thập dữ liệu

Yêu cầu:

- Sinh viên tự crawl dữ liệu.
- Số lượng ảnh hợp lệ sau khi làm sạch phải lớn hơn 10.000 mẫu.
- Có dẫn nguồn dữ liệu.
- Có mô tả cách thức thu thập.
- Có thống kê mô tả và trực quan hóa dữ liệu.

Đầu ra cần có:

- Script crawl dữ liệu.
- File metadata lưu thông tin ảnh đã crawl.
- Dataset đã tổ chức theo class.
- Script kiểm tra và làm sạch ảnh.
- Bảng thống kê số lượng ảnh theo class.
- Biểu đồ phân bố dữ liệu.
- Một số ảnh mẫu minh họa từng class.

## 2.2 Trích xuất đặc trưng

Yêu cầu:

- Trình bày cách lựa chọn đặc trưng.
- Trình bày quá trình làm sạch và chuẩn hóa dữ liệu.
- Có giảm chiều dữ liệu.
- Có trực quan hóa kết quả của các quá trình xử lý đặc trưng.

Đầu ra cần có:

- Pipeline preprocessing ảnh.
- Pipeline augmentation ảnh.
- Train/Validation/Test split.
- Feature vector trích xuất từ ảnh hoặc từ backbone CNN.
- PCA visualization.
- t-SNE hoặc UMAP visualization.
- Biểu đồ minh họa ảnh trước/sau preprocessing.
- Biểu đồ minh họa ảnh trước/sau augmentation.

## 2.3 Mô hình hóa dữ liệu

Yêu cầu:

- Chọn ít nhất 2 mô hình hoặc thuật toán phù hợp.
- Chia dữ liệu thành Train/Validation/Test theo tỉ lệ hợp lý.
- Trình bày đồ thị thể hiện hiệu quả mô hình trong quá trình huấn luyện, hiệu chỉnh và kiểm thử.
- Accuracy trên test set không nhỏ hơn 85%.

Đầu ra cần có:

- Mô hình MobileNetV2 đã train.
- Mô hình ResNet50 đã train.
- Biểu đồ training loss/validation loss.
- Biểu đồ training accuracy/validation accuracy.
- Confusion matrix.
- Classification report gồm precision, recall, F1-score.
- Bảng so sánh accuracy, inference time, FPS, model size.
- Nhận xét mô hình tốt hơn theo từng tiêu chí.

---

# 3. Nguyên tắc phân công

Nhóm có 4 người và thời gian thực hiện là 3 tuần.

Nguyên tắc chia việc:

- Mỗi người có một phần kỹ thuật chính, đủ rõ để bảo vệ riêng.
- Không giao toàn bộ phần khó cho một người duy nhất.
- Các phần có phụ thuộc được xếp theo thứ tự: dữ liệu -> đặc trưng -> mô hình -> đánh giá.
- Mỗi người đều có deliverables kiểm tra được bằng file, script, biểu đồ hoặc kết quả thực nghiệm.
- Báo cáo, slide và phụ lục không tính vào workload kỹ thuật chính.

Lưu ý quan trọng:

```text
Phần viết báo cáo không được tính là công việc kỹ thuật chính.
Mỗi thành viên vẫn phải có đóng góp kỹ thuật rõ ràng trong source code, dataset, mô hình hoặc kết quả đánh giá.
```

---

# 4. Phân vai tổng quan

| Thành viên | Vai trò chính | Phần phụ trách |
|---|---|---|
| Người 1 | Data Collection Engineer | Crawl dữ liệu, nguồn dữ liệu, metadata, tổ chức dataset |
| Người 2 | Data Processing & Feature Engineer | Làm sạch, preprocessing, augmentation, split, PCA/t-SNE |
| Người 3 | MobileNetV2 Engineer | Train, fine-tune, đánh giá MobileNetV2 |
| Người 4 | ResNet50 & Evaluation Engineer | Train ResNet50, benchmark, so sánh hai mô hình |

---

# 5. Workflow tổng thể

```text
Xác định class biển báo
        ↓
Crawl dữ liệu từ nhiều nguồn
        ↓
Lưu metadata và dẫn nguồn
        ↓
Làm sạch dữ liệu
        ↓
Thống kê và trực quan hóa dữ liệu
        ↓
Preprocessing và augmentation
        ↓
Chia train/validation/test
        ↓
Trích xuất đặc trưng
        ↓
PCA/t-SNE visualization
        ↓
Train MobileNetV2
        ↓
Train ResNet50
        ↓
Fine-tuning
        ↓
Đánh giá trên test set
        ↓
Benchmark và so sánh mô hình
```

---

# 6. Danh sách class đề xuất

Để đạt hơn 10.000 ảnh và vẫn đảm bảo khả năng phân loại tốt trong 3 tuần, nên chọn khoảng 8 đến 12 class phổ biến.

Danh sách class đề xuất:

| STT | Class | Tên tiếng Việt | Ghi chú |
|---|---|---|---|
| 1 | stop | Biển báo dừng | Dễ crawl, nhiều ảnh |
| 2 | no_entry | Cấm vào | Dễ phân biệt |
| 3 | speed_limit | Giới hạn tốc độ | Có nhiều biến thể |
| 4 | pedestrian_crossing | Người đi bộ qua đường | Phổ biến |
| 5 | traffic_light | Đèn giao thông | Dễ có nhiễu nền |
| 6 | yield | Nhường đường | Dễ phân biệt |
| 7 | no_parking | Cấm đỗ xe | Cần kiểm soát biến thể |
| 8 | warning | Cảnh báo nguy hiểm | Class rộng, cần định nghĩa rõ |
| 9 | turn_left | Rẽ trái | Có thể bổ sung |
| 10 | turn_right | Rẽ phải | Có thể bổ sung |

Khuyến nghị:

- Nếu thời gian crawl bị chậm, dùng 8 class chính.
- Nếu dữ liệu mỗi class cân bằng tốt, dùng 10 class.
- Mỗi class nên có tối thiểu 1.000 ảnh thô trước khi clean.
- Sau clean, tổng dataset hợp lệ phải còn lớn hơn 10.000 ảnh.

---

# 7. Nguồn dữ liệu và quy định crawl

## 7.1 Nguồn crawl chính

Các nguồn nên dùng:

- Google Images thông qua Selenium hoặc thư viện crawl ảnh phù hợp.
- Bing Images thông qua Selenium hoặc API nếu có.
- Wikimedia Commons.
- Flickr.
- Open Images Dataset thông qua URL ảnh và nhãn liên quan.
- Website giao thông, an toàn đường bộ, thư viện ảnh biển báo công khai.

## 7.2 Nguồn không nên tính là crawl chính

Không nên tính các dataset tải sẵn từ Kaggle là dữ liệu tự crawl chính.

Kaggle chỉ nên dùng cho:

- Tham khảo class.
- Tham khảo cách tổ chức dataset.
- Đối chiếu kết quả.
- Không tính vào mốc hơn 10.000 ảnh tự crawl nếu giảng viên yêu cầu nghiêm ngặt.

## 7.3 Metadata bắt buộc

Mỗi ảnh crawl nên lưu metadata:

| Trường | Ý nghĩa |
|---|---|
| image_id | ID ảnh |
| class_name | Tên class |
| source_url | URL ảnh hoặc trang chứa ảnh |
| source_site | Website nguồn |
| query | Từ khóa crawl |
| crawl_date | Ngày crawl |
| raw_path | Đường dẫn ảnh thô |
| clean_path | Đường dẫn ảnh sau clean |
| width | Chiều rộng ảnh |
| height | Chiều cao ảnh |
| status | valid, duplicate, corrupted, blurry, removed |

File metadata đề xuất:

```text
data/metadata/images_metadata.csv
```

---

# 8. Cấu trúc thư mục đề xuất

```text
project/
│
├── data/
│   ├── raw/
│   │   ├── stop/
│   │   ├── no_entry/
│   │   ├── speed_limit/
│   │   └── ...
│   │
│   ├── clean/
│   │   ├── stop/
│   │   ├── no_entry/
│   │   ├── speed_limit/
│   │   └── ...
│   │
│   ├── split/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── metadata/
│       ├── images_metadata.csv
│       └── data_summary.csv
│
├── notebooks/
│   ├── 01_data_statistics.ipynb
│   ├── 02_feature_visualization.ipynb
│   ├── 03_mobilenetv2_training.ipynb
│   └── 04_resnet50_training.ipynb
│
├── src/
│   ├── crawl/
│   │   ├── crawl_images.py
│   │   └── crawl_config.yaml
│   │
│   ├── data/
│   │   ├── clean_images.py
│   │   ├── remove_duplicates.py
│   │   ├── split_dataset.py
│   │   └── data_statistics.py
│   │
│   ├── features/
│   │   ├── preprocessing.py
│   │   ├── augmentation.py
│   │   ├── extract_features.py
│   │   └── visualize_features.py
│   │
│   ├── models/
│   │   ├── train_mobilenetv2.py
│   │   ├── train_resnet50.py
│   │   └── model_utils.py
│   │
│   └── evaluation/
│       ├── evaluate_model.py
│       ├── benchmark.py
│       └── compare_models.py
│
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── logs/
│   └── metrics/
│
├── requirements.txt
└── README.md
```

---

# 9. Chi tiết công việc Người 1
# Data Collection Engineer

---

## 9.1 Trách nhiệm chính

Người 1 phụ trách phần thu thập dữ liệu.

Công việc gồm:

- Xác định danh sách class cuối cùng.
- Tạo danh sách từ khóa crawl cho từng class.
- Crawl ảnh từ nhiều nguồn.
- Lưu ảnh thô theo đúng cấu trúc thư mục.
- Lưu metadata cho từng ảnh.
- Ghi lại nguồn dữ liệu và cách thức thu thập.
- Kiểm tra sơ bộ số lượng ảnh theo class.

## 9.2 Công việc chi tiết

### 9.2.1 Xác định class

Người 1 phối hợp với cả nhóm để chốt:

- Số class chính thức.
- Tên class dùng trong code.
- Tên tiếng Việt dùng khi trình bày.
- Số lượng ảnh mục tiêu mỗi class.

Ví dụ mục tiêu nếu chọn 10 class:

| Class | Số ảnh thô mục tiêu | Số ảnh sạch kỳ vọng |
|---|---:|---:|
| stop | 1.300 | 1.000+ |
| no_entry | 1.300 | 1.000+ |
| speed_limit | 1.500 | 1.100+ |
| pedestrian_crossing | 1.300 | 1.000+ |
| traffic_light | 1.300 | 1.000+ |
| yield | 1.300 | 1.000+ |
| no_parking | 1.300 | 1.000+ |
| warning | 1.500 | 1.100+ |
| turn_left | 1.200 | 900+ |
| turn_right | 1.200 | 900+ |

Mục tiêu tổng:

```text
Ảnh thô: 13.000 - 14.000 ảnh
Ảnh sạch sau lọc: > 10.000 ảnh
```

### 9.2.2 Từ khóa crawl

Mỗi class cần có nhiều từ khóa để tăng độ đa dạng.

Ví dụ:

| Class | Từ khóa tiếng Anh | Từ khóa tiếng Việt |
|---|---|---|
| stop | stop sign, road stop sign, traffic stop sign | biển báo dừng |
| no_entry | no entry sign, do not enter sign | biển cấm vào |
| speed_limit | speed limit sign, 30 speed sign, 50 speed sign | biển giới hạn tốc độ |
| pedestrian_crossing | pedestrian crossing sign, crosswalk sign | biển người đi bộ |
| yield | yield sign, give way sign | biển nhường đường |
| no_parking | no parking sign | biển cấm đỗ xe |

### 9.2.3 Script crawl

Script cần hỗ trợ:

- Nhập class name.
- Nhập danh sách query.
- Giới hạn số ảnh mỗi query.
- Tải ảnh về thư mục `data/raw/<class_name>/`.
- Lưu metadata vào CSV.
- Bỏ qua URL trùng.
- Log lỗi khi ảnh không tải được.

File đề xuất:

```text
src/crawl/crawl_images.py
src/crawl/crawl_config.yaml
```

### 9.2.4 Metadata và dẫn nguồn

Người 1 cần đảm bảo có bằng chứng tự crawl:

- Danh sách URL nguồn.
- Từ khóa dùng để crawl.
- Ngày crawl.
- Số lượng ảnh crawl từ từng nguồn.
- Số lượng ảnh hợp lệ sau lọc.

Biểu đồ cần bàn giao:

- Số ảnh crawl theo nguồn.
- Số ảnh crawl theo class.
- Tỉ lệ ảnh hợp lệ/lỗi/trùng/mờ.

## 9.3 Deliverables của Người 1

Người 1 bàn giao:

- `src/crawl/crawl_images.py`
- `src/crawl/crawl_config.yaml`
- `data/raw/`
- `data/metadata/images_metadata.csv`
- Bảng số lượng ảnh thô theo class.
- Bảng nguồn dữ liệu.
- Biểu đồ thống kê nguồn crawl.
- Mô tả quy trình crawl.

## 9.4 Tiêu chí hoàn thành

Người 1 được xem là hoàn thành khi:

- Có hơn 13.000 ảnh thô hoặc đủ để sau clean còn hơn 10.000 ảnh.
- Mỗi ảnh có metadata nguồn.
- Dataset raw được chia đúng thư mục class.
- Có thể chạy lại script crawl ở mức cơ bản.
- Có thống kê số lượng ảnh theo class và theo nguồn.

---

# 10. Chi tiết công việc Người 2
# Data Processing & Feature Engineer

---

## 10.1 Trách nhiệm chính

Người 2 phụ trách xử lý dữ liệu và trích xuất đặc trưng.

Công việc gồm:

- Làm sạch ảnh.
- Loại ảnh lỗi, trùng, quá mờ hoặc sai định dạng.
- Chuẩn hóa kích thước ảnh.
- Chuẩn hóa pixel.
- Chia Train/Validation/Test.
- Thiết kế augmentation.
- Trích xuất feature vector.
- Giảm chiều bằng PCA và t-SNE/UMAP.
- Trực quan hóa quá trình xử lý dữ liệu và đặc trưng.

## 10.2 Làm sạch dữ liệu

Các bước làm sạch:

| Bước | Phương pháp | Mục tiêu |
|---|---|---|
| Kiểm tra ảnh lỗi | PIL/OpenCV verify | Loại ảnh không đọc được |
| Kiểm tra kích thước | width, height | Loại ảnh quá nhỏ |
| Kiểm tra trùng lặp | perceptual hash | Loại ảnh giống nhau |
| Kiểm tra ảnh mờ | variance of Laplacian | Loại ảnh chất lượng thấp |
| Kiểm tra định dạng | jpg, jpeg, png, webp | Chuẩn hóa định dạng |
| Kiểm tra class imbalance | count per class | Tránh lệch class quá mạnh |

Ngưỡng đề xuất:

```text
Kích thước tối thiểu: 64 x 64
Blur threshold: tùy thử nghiệm, ghi rõ trong kết quả
Duplicate threshold: dựa trên perceptual hash distance
```

## 10.3 Preprocessing

Ảnh đầu vào cho MobileNetV2 và ResNet50:

```text
224 x 224 x 3
```

Các bước:

- Resize ảnh về 224x224.
- Convert RGB.
- Normalize pixel.
- Dùng `preprocess_input()` tương ứng với backbone nếu dùng Keras/TensorFlow.
- Lưu pipeline dùng chung cho cả hai mô hình.

File đề xuất:

```text
src/features/preprocessing.py
```

## 10.4 Data augmentation

Augmentation dùng cho train set, không dùng cho validation/test.

Kỹ thuật đề xuất:

- Random rotation nhỏ.
- Random zoom.
- Brightness adjustment.
- Width shift.
- Height shift.
- Random contrast.

Không nên augmentation quá mạnh vì có thể làm sai ý nghĩa biển báo.

Ví dụ:

```python
rotation_range = 10-15
zoom_range = 0.1-0.15
brightness_range = [0.8, 1.2]
width_shift_range = 0.1
height_shift_range = 0.1
```

## 10.5 Chia dữ liệu

Tỉ lệ đề xuất:

| Tập dữ liệu | Tỉ lệ | Mục đích |
|---|---:|---|
| Train | 70% | Huấn luyện |
| Validation | 15% | Hiệu chỉnh hyperparameter |
| Test | 15% | Đánh giá cuối cùng |

Yêu cầu:

- Split theo kiểu stratified để giữ tỉ lệ class.
- Không để ảnh trùng hoặc gần trùng xuất hiện ở nhiều tập.
- Test set chỉ dùng một lần cho đánh giá cuối.

File đề xuất:

```text
src/data/split_dataset.py
```

## 10.6 Trích xuất đặc trưng

Vì bài toán là ảnh, đặc trưng có thể trình bày theo 2 cấp:

### 10.6.1 Đặc trưng mức ảnh

Gồm:

- Kích thước ảnh.
- Phân bố màu RGB.
- Độ sáng.
- Độ tương phản.
- Độ sắc nét.

Mục đích:

- Mô tả dữ liệu.
- Phát hiện ảnh bất thường.
- Hỗ trợ giải thích bước cleaning.

### 10.6.2 Đặc trưng học sâu

Trích xuất feature vector từ backbone CNN:

- MobileNetV2 bỏ classification head.
- ResNet50 bỏ classification head.
- Dùng GlobalAveragePooling để lấy embedding.

Mục đích:

- Trực quan hóa feature space.
- So sánh khả năng phân tách class trước/sau fine-tuning.
- Dùng PCA/t-SNE để minh họa cụm class.

File đề xuất:

```text
src/features/extract_features.py
src/features/visualize_features.py
```

## 10.7 Giảm chiều và trực quan hóa

Cần có:

- PCA 2D plot.
- t-SNE hoặc UMAP 2D plot.
- Biểu đồ explained variance của PCA.
- Sample ảnh theo từng class.
- Sample ảnh bị loại khi cleaning.
- Sample ảnh sau augmentation.

## 10.8 Deliverables của Người 2

Người 2 bàn giao:

- `src/data/clean_images.py`
- `src/data/remove_duplicates.py`
- `src/data/split_dataset.py`
- `src/features/preprocessing.py`
- `src/features/augmentation.py`
- `src/features/extract_features.py`
- `src/features/visualize_features.py`
- `data/clean/`
- `data/split/`
- PCA plot.
- t-SNE/UMAP plot.
- Biểu đồ ảnh trước/sau augmentation.
- Bảng thống kê ảnh bị loại.

## 10.9 Tiêu chí hoàn thành

Người 2 được xem là hoàn thành khi:

- Dataset sạch còn hơn 10.000 ảnh.
- Có Train/Validation/Test split đúng tỉ lệ.
- Không có ảnh lỗi trong dataset sạch.
- Có visualization dữ liệu và đặc trưng.
- Có pipeline preprocessing dùng được cho cả hai mô hình.

---

# 11. Chi tiết công việc Người 3
# MobileNetV2 Engineer

---

## 11.1 Trách nhiệm chính

Người 3 phụ trách mô hình MobileNetV2.

Công việc gồm:

- Xây dựng kiến trúc MobileNetV2 transfer learning.
- Train baseline model.
- Fine-tune model.
- Ghi log quá trình huấn luyện.
- Đánh giá trên validation và test set.
- Xuất biểu đồ training.
- Xuất confusion matrix và classification report cho MobileNetV2.

## 11.2 Kiến trúc mô hình

Mô hình đề xuất:

```text
Input: 224 x 224 x 3
Backbone: MobileNetV2 pretrained ImageNet
Pooling: GlobalAveragePooling2D
Regularization: Dropout
Classifier: Dense(num_classes, activation='softmax')
```

Chiến lược train:

Giai đoạn 1:

- Freeze backbone.
- Train classification head.
- Learning rate tương đối lớn hơn.

Giai đoạn 2:

- Unfreeze một phần các layer cuối.
- Fine-tune với learning rate nhỏ.
- Dùng EarlyStopping và ModelCheckpoint.

## 11.3 Hyperparameter cần thử

Các cấu hình nên thử:

| Tham số | Giá trị đề xuất |
|---|---|
| image size | 224x224 |
| batch size | 16 hoặc 32 |
| optimizer | Adam |
| learning rate head | 1e-3 |
| learning rate fine-tune | 1e-5 đến 1e-4 |
| dropout | 0.3 đến 0.5 |
| loss | categorical crossentropy |
| epochs baseline | 10-15 |
| epochs fine-tune | 10-20 |

## 11.4 Chỉ số đánh giá

MobileNetV2 cần có:

- Train accuracy.
- Validation accuracy.
- Test accuracy.
- Train loss.
- Validation loss.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- Inference time.
- Model size.

## 11.5 Biểu đồ cần xuất

- Training/validation accuracy theo epoch.
- Training/validation loss theo epoch.
- Confusion matrix.
- Bar chart precision/recall/F1 theo class.

## 11.6 Deliverables của Người 3

Người 3 bàn giao:

- `src/models/train_mobilenetv2.py`
- File model tốt nhất trong `outputs/models/`
- Log huấn luyện trong `outputs/logs/`
- Metrics MobileNetV2 trong `outputs/metrics/`
- Biểu đồ training MobileNetV2 trong `outputs/figures/`
- Confusion matrix MobileNetV2.
- Nhận xét kỹ thuật về MobileNetV2.

## 11.7 Tiêu chí hoàn thành

Người 3 được xem là hoàn thành khi:

- MobileNetV2 train được end-to-end.
- Có model checkpoint tốt nhất.
- Có đầy đủ biểu đồ loss/accuracy.
- Có test accuracy và classification report.
- Accuracy MobileNetV2 nên đạt tối thiểu 85%, hoặc nếu chưa đạt phải có phân tích nguyên nhân và hướng cải thiện.

---

# 12. Chi tiết công việc Người 4
# ResNet50 & Evaluation Engineer

---

## 12.1 Trách nhiệm chính

Người 4 phụ trách ResNet50 và đánh giá tổng hợp.

Công việc gồm:

- Xây dựng kiến trúc ResNet50 transfer learning.
- Train baseline ResNet50.
- Fine-tune ResNet50.
- Đánh giá ResNet50 trên test set.
- Benchmark MobileNetV2 và ResNet50 theo cùng điều kiện.
- So sánh hai mô hình bằng bảng và biểu đồ.
- Đưa ra kết luận kỹ thuật.

## 12.2 Kiến trúc mô hình

Mô hình đề xuất:

```text
Input: 224 x 224 x 3
Backbone: ResNet50 pretrained ImageNet
Pooling: GlobalAveragePooling2D
Regularization: Dropout
Classifier: Dense(num_classes, activation='softmax')
```

Chiến lược train:

Giai đoạn 1:

- Freeze backbone.
- Train classification head.

Giai đoạn 2:

- Fine-tune một số block cuối của ResNet50.
- Dùng learning rate nhỏ.
- Theo dõi overfitting vì ResNet50 nặng hơn MobileNetV2.

## 12.3 Hyperparameter cần thử

| Tham số | Giá trị đề xuất |
|---|---|
| image size | 224x224 |
| batch size | 16 hoặc 32 |
| optimizer | Adam hoặc SGD momentum |
| learning rate head | 1e-3 |
| learning rate fine-tune | 1e-5 đến 1e-4 |
| dropout | 0.4 đến 0.5 |
| loss | categorical crossentropy |
| epochs baseline | 10-15 |
| epochs fine-tune | 10-20 |

## 12.4 Benchmark

Benchmark phải chạy cùng điều kiện:

- Cùng test set.
- Cùng image size.
- Cùng thiết bị đo nếu có thể.
- Cùng batch size khi đo inference.

Chỉ số benchmark:

| Metric | Ý nghĩa |
|---|---|
| Test accuracy | Độ chính xác trên test set |
| Precision | Độ chính xác theo dự đoán dương |
| Recall | Khả năng phát hiện đúng từng class |
| F1-score | Cân bằng precision/recall |
| Inference time | Thời gian dự đoán trung bình |
| FPS | Số ảnh xử lý mỗi giây |
| Model size | Dung lượng model |
| Number of parameters | Số tham số |

## 12.5 Bảng so sánh bắt buộc

| Metric | MobileNetV2 | ResNet50 | Nhận xét |
|---|---:|---:|---|
| Test accuracy | | | |
| Macro precision | | | |
| Macro recall | | | |
| Macro F1-score | | | |
| Inference time/image | | | |
| FPS | | | |
| Model size | | | |
| Parameters | | | |

## 12.6 Kết luận kỹ thuật cần có

Người 4 cần kết luận:

- Mô hình nào có accuracy cao hơn.
- Mô hình nào nhanh hơn.
- Mô hình nào nhẹ hơn.
- Mô hình nào phù hợp nếu triển khai realtime.
- Mô hình nào phù hợp nếu ưu tiên độ chính xác.
- Class nào dễ nhầm lẫn nhất và lý do có thể.
- Accuracy đã đạt yêu cầu 85% hay chưa.

## 12.7 Deliverables của Người 4

Người 4 bàn giao:

- `src/models/train_resnet50.py`
- `src/evaluation/evaluate_model.py`
- `src/evaluation/benchmark.py`
- `src/evaluation/compare_models.py`
- File model ResNet50 tốt nhất.
- Metrics ResNet50.
- Biểu đồ training ResNet50.
- Confusion matrix ResNet50.
- Bảng benchmark hai mô hình.
- Biểu đồ so sánh hai mô hình.
- Kết luận kỹ thuật.

## 12.8 Tiêu chí hoàn thành

Người 4 được xem là hoàn thành khi:

- ResNet50 train được end-to-end.
- Có kết quả test set đầy đủ.
- Có benchmark công bằng giữa hai mô hình.
- Có bảng và biểu đồ so sánh.
- Có kết luận rõ mô hình nào phù hợp hơn.

---

# 13. Timeline 3 tuần

Thời gian thực hiện kỹ thuật là 3 tuần.

## 13.1 Tuần 1: Dữ liệu và làm sạch

Mục tiêu tuần 1:

- Hoàn thành crawl dữ liệu.
- Dataset thô đạt khoảng 13.000 đến 14.000 ảnh.
- Dataset sạch còn hơn 10.000 ảnh.
- Có thống kê mô tả dữ liệu.

| Ngày | Người 1 | Người 2 | Người 3 | Người 4 |
|---|---|---|---|---|
| Ngày 1 | Chốt class, tạo query crawl | Thiết kế tiêu chí clean | Chuẩn bị môi trường train | Chuẩn bị môi trường train/eval |
| Ngày 2 | Crawl batch 1 | Viết script kiểm tra ảnh lỗi | Tạo skeleton MobileNetV2 | Tạo skeleton ResNet50 |
| Ngày 3 | Crawl batch 2 | Viết duplicate/blur detection | Test dataloader mẫu | Test dataloader mẫu |
| Ngày 4 | Crawl bổ sung class thiếu | Clean dữ liệu lần 1 | Kiểm tra preprocessing input | Kiểm tra preprocessing input |
| Ngày 5 | Hoàn thiện metadata nguồn | Clean dữ liệu lần 2 | Chuẩn bị config training | Chuẩn bị config training |
| Ngày 6 | Thống kê ảnh theo nguồn/class | Split train/val/test | Dry-run MobileNetV2 ít epoch | Dry-run ResNet50 ít epoch |
| Ngày 7 | Chốt dataset raw | Chốt dataset clean + split | Review chất lượng dữ liệu | Review chất lượng dữ liệu |

Deliverables cuối tuần 1:

- Dataset raw.
- Dataset clean.
- Metadata có nguồn.
- Train/Validation/Test split.
- Biểu đồ thống kê dữ liệu.

## 13.2 Tuần 2: Đặc trưng và train baseline

Mục tiêu tuần 2:

- Hoàn thiện preprocessing và augmentation.
- Có PCA/t-SNE visualization.
- Train baseline MobileNetV2 và ResNet50.
- Có kết quả validation đầu tiên.

| Ngày | Người 1 | Người 2 | Người 3 | Người 4 |
|---|---|---|---|---|
| Ngày 8 | Bổ sung dữ liệu class thiếu nếu cần | Hoàn thiện preprocessing | Train MobileNetV2 baseline | Train ResNet50 baseline |
| Ngày 9 | Kiểm tra lại metadata | Hoàn thiện augmentation | Theo dõi loss/accuracy | Theo dõi loss/accuracy |
| Ngày 10 | Tạo biểu đồ nguồn dữ liệu | Trích xuất feature vector | Tune learning rate | Tune learning rate |
| Ngày 11 | Tạo sample image grid | PCA visualization | Train baseline lần 2 | Train baseline lần 2 |
| Ngày 12 | Hỗ trợ kiểm tra class imbalance | t-SNE/UMAP visualization | Đánh giá validation | Đánh giá validation |
| Ngày 13 | Review dữ liệu sai nhãn | Augmentation experiment | Chuẩn bị fine-tune | Chuẩn bị fine-tune |
| Ngày 14 | Chốt thống kê dữ liệu | Chốt biểu đồ đặc trưng | Chốt baseline MobileNetV2 | Chốt baseline ResNet50 |

Deliverables cuối tuần 2:

- Preprocessing pipeline.
- Augmentation pipeline.
- PCA/t-SNE visualization.
- Baseline MobileNetV2.
- Baseline ResNet50.
- Validation metrics ban đầu.

## 13.3 Tuần 3: Fine-tuning, đánh giá và benchmark

Mục tiêu tuần 3:

- Fine-tune hai mô hình.
- Đạt accuracy test set không nhỏ hơn 85%.
- Hoàn thiện đánh giá và benchmark.
- Có đầy đủ biểu đồ kỹ thuật.

| Ngày | Người 1 | Người 2 | Người 3 | Người 4 |
|---|---|---|---|---|
| Ngày 15 | Kiểm tra lại class thiếu/sai | Hỗ trợ augmentation tốt nhất | Fine-tune MobileNetV2 | Fine-tune ResNet50 |
| Ngày 16 | Bổ sung dữ liệu nếu accuracy thấp | Phân tích class imbalance | Fine-tune lần 2 | Fine-tune lần 2 |
| Ngày 17 | Kiểm tra mẫu bị nhầm | Trích xuất feature sau fine-tune | Test MobileNetV2 | Test ResNet50 |
| Ngày 18 | Hỗ trợ sửa nhãn sai | Vẽ PCA/t-SNE sau fine-tune | Confusion matrix MobileNetV2 | Confusion matrix ResNet50 |
| Ngày 19 | Chốt data evidence | Chốt feature figures | Xuất metrics MobileNetV2 | Xuất metrics ResNet50 |
| Ngày 20 | Review checklist dữ liệu | Review checklist đặc trưng | Hỗ trợ benchmark | Benchmark + compare models |
| Ngày 21 | Chốt phần dữ liệu | Chốt phần đặc trưng | Chốt MobileNetV2 | Chốt so sánh cuối |

Deliverables cuối tuần 3:

- Model MobileNetV2 cuối.
- Model ResNet50 cuối.
- Test metrics đầy đủ.
- Confusion matrix hai mô hình.
- Biểu đồ loss/accuracy hai mô hình.
- Benchmark hai mô hình.
- Bảng so sánh kỹ thuật.

---

# 14. Checklist yêu cầu đề bài

| Yêu cầu đề bài | Cách đáp ứng | Người phụ trách chính |
|---|---|---|
| SV tự thu thập dữ liệu | Tự crawl bằng script, lưu URL nguồn | Người 1 |
| Số lượng >10.000 mẫu | Dataset sạch sau lọc >10.000 ảnh | Người 1, Người 2 |
| Dẫn nguồn dữ liệu | Metadata có source_url/source_site | Người 1 |
| Mô tả cách thức thu thập | Có script, config, mô tả query và nguồn | Người 1 |
| Thống kê mô tả trực quan | Bar chart, histogram, sample grid | Người 1, Người 2 |
| Lựa chọn đặc trưng | Đặc trưng ảnh + embedding CNN | Người 2 |
| Làm sạch dữ liệu | Remove corrupt/duplicate/blur | Người 2 |
| Chuẩn hóa dữ liệu | Resize, RGB, normalize, preprocess_input | Người 2 |
| Giảm chiều | PCA, t-SNE hoặc UMAP | Người 2 |
| Trực quan hóa đặc trưng | PCA/t-SNE plot, feature visualization | Người 2 |
| Ít nhất 2 mô hình | MobileNetV2 và ResNet50 | Người 3, Người 4 |
| Train/Validation/Test | Split 70/15/15 stratified | Người 2 |
| Đồ thị huấn luyện | Loss/accuracy curves | Người 3, Người 4 |
| Đồ thị kiểm thử | Confusion matrix, metric charts | Người 3, Người 4 |
| Accuracy >=85% | Fine-tuning và đánh giá test set | Người 3, Người 4 |
| Báo cáo >=35 trang | Không tính vào workload kỹ thuật | Cả nhóm xử lý riêng |

---

# 15. Tiêu chí nghiệm thu nội bộ

Trước khi kết thúc project, nhóm cần kiểm tra:

## 15.1 Dataset

- Tổng số ảnh sạch lớn hơn 10.000.
- Có ít nhất 8 class.
- Không class nào quá ít dữ liệu.
- Có metadata nguồn cho ảnh.
- Có thống kê ảnh raw, ảnh removed, ảnh clean.
- Có biểu đồ phân bố class.
- Có ví dụ ảnh từng class.

## 15.2 Feature engineering

- Có preprocessing pipeline rõ ràng.
- Có augmentation chỉ áp dụng cho train set.
- Có Train/Validation/Test split.
- Có PCA visualization.
- Có t-SNE hoặc UMAP visualization.
- Có nhận xét về khả năng phân tách class.

## 15.3 Modeling

- Có 2 mô hình train end-to-end.
- Có log training.
- Có biểu đồ loss/accuracy.
- Có test accuracy.
- Có precision/recall/F1-score.
- Có confusion matrix.
- Có benchmark tốc độ và dung lượng model.
- Ít nhất một mô hình đạt accuracy >= 85%.

Khuyến nghị tốt hơn:

```text
Cả hai mô hình đều nên đạt >= 85%.
Nếu chỉ một mô hình đạt >= 85%, cần giải thích rõ vì sao mô hình còn lại thấp hơn.
```

---

# 16. Phương án nếu accuracy dưới 85%

Nếu accuracy chưa đạt yêu cầu, nhóm thực hiện theo thứ tự:

1. Kiểm tra lại dữ liệu sai nhãn.
2. Kiểm tra class imbalance.
3. Bổ sung dữ liệu cho class yếu.
4. Điều chỉnh augmentation.
5. Tăng số epoch kèm EarlyStopping.
6. Fine-tune thêm layer cuối của backbone.
7. Giảm learning rate khi fine-tune.
8. Thử dropout khác nhau.
9. Kiểm tra confusion matrix để tìm class hay nhầm.
10. Gộp hoặc định nghĩa lại class quá mơ hồ nếu hợp lý.

Không nên:

- Đánh giá bằng validation set thay cho test set.
- Dùng ảnh test trong training.
- Augment validation/test set.
- Báo accuracy cao nhưng không có confusion matrix hoặc classification report.

---

# 17. Quy tắc làm việc nhóm

## 17.1 Quy tắc Git

Mỗi người làm trên branch riêng:

| Người | Branch đề xuất |
|---|---|
| Người 1 | `feature/data-crawling` |
| Người 2 | `feature/data-processing` |
| Người 3 | `feature/mobilenetv2` |
| Người 4 | `feature/resnet50-evaluation` |

Quy tắc commit:

- Commit nhỏ theo từng chức năng.
- Không commit dataset quá lớn nếu repo không dùng Git LFS.
- Không commit file tạm, cache, checkpoint quá nặng.
- Metrics, biểu đồ và config quan trọng thì nên lưu.

## 17.2 Quy tắc đặt tên file

Tên class:

```text
lowercase_with_underscore
```

Tên ảnh:

```text
<class_name>_<source>_<index>.jpg
```

Ví dụ:

```text
stop_google_000001.jpg
no_entry_wikimedia_000125.jpg
```

## 17.3 Quy tắc thực nghiệm

Mỗi lần train cần lưu:

- Model name.
- Dataset version.
- Số class.
- Số ảnh train/val/test.
- Image size.
- Batch size.
- Learning rate.
- Optimizer.
- Epochs.
- Augmentation config.
- Accuracy tốt nhất.
- Test metrics.

File log đề xuất:

```text
outputs/metrics/experiment_log.csv
```

---

# 18. Phân chia khi bảo vệ

Mỗi người trình bày đúng phần kỹ thuật mình làm.

| Thành viên | Nội dung trình bày |
|---|---|
| Người 1 | Nguồn dữ liệu, cách crawl, metadata, số lượng ảnh, bằng chứng tự thu thập |
| Người 2 | Làm sạch dữ liệu, preprocessing, augmentation, split, PCA/t-SNE |
| Người 3 | MobileNetV2, transfer learning, fine-tuning, kết quả đánh giá |
| Người 4 | ResNet50, benchmark, so sánh mô hình, kết luận kỹ thuật |

Gợi ý demo:

- Hiển thị vài ảnh sample theo class.
- Hiển thị biểu đồ phân bố dataset.
- Hiển thị PCA/t-SNE feature space.
- Hiển thị training curves.
- Hiển thị confusion matrix.
- Chạy thử dự đoán một ảnh mới.
- So sánh MobileNetV2 và ResNet50.

---

# 19. Bảng tổng hợp workload

| Thành viên | Data  | Cleaning | Feature | Model       | Evaluation | Benchmark | Visualization |
|------------|-------|----------|---------|-------------|------------|-----------|---------------|
| Người 1    | Chính | Phụ      | Không   | Không       | Không      | Không     | Phụ           |
| Người 2    | Phụ   | Chính    | Chính   | Phụ         | Phụ        | Không     | Chính         |
| Người 3    | Không | Không    | Phụ     | MobileNetV2 | Chính cho MobileNetV2 | Phụ | Chính cho MobileNetV2 |
| Người 4    | Không | Không    | Phụ     | ResNet50    | Chính cho ResNet50 | Chính | Chính cho so sánh |

Nhận xét:

- Người 1 có phần kỹ thuật rõ ràng ở dữ liệu và crawling.
- Người 2 có phần kỹ thuật rõ ràng ở xử lý dữ liệu và feature engineering.
- Người 3 có phần kỹ thuật rõ ràng ở mô hình MobileNetV2.
- Người 4 có phần kỹ thuật rõ ràng ở mô hình ResNet50 và benchmark.
- Workload được chia theo pipeline, tránh trùng việc nhưng vẫn có điểm phối hợp.

---

# 20. Kết luận

Bản phân công này phù hợp với:

- Nhóm 4 người.
- Thời gian kỹ thuật 3 tuần.
- Yêu cầu tự crawl hơn 10.000 mẫu.
- Yêu cầu có thống kê và trực quan hóa dữ liệu.
- Yêu cầu trích xuất đặc trưng, chuẩn hóa, giảm chiều.
- Yêu cầu có ít nhất 2 mô hình.
- Yêu cầu Train/Validation/Test split.
- Yêu cầu có đồ thị huấn luyện, hiệu chỉnh và kiểm thử.
- Yêu cầu accuracy không nhỏ hơn 85%.

Phần báo cáo tối thiểu 35 trang, slide bảo vệ, phụ lục và tài liệu tham khảo sẽ được xử lý riêng, không tính là workload kỹ thuật chính của từng thành viên.
