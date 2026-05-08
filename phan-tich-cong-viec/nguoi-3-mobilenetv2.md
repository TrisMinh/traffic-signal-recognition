# PHÂN TÍCH CÔNG VIỆC NGƯỜI 3
# Vai trò: MobileNetV2 Engineer

---

# 1. Mục tiêu phần việc

Người 3 chịu trách nhiệm chính cho mô hình **MobileNetV2**.

Mục tiêu:

- Xây dựng mô hình phân loại biển báo giao thông bằng MobileNetV2.
- Sử dụng transfer learning từ ImageNet.
- Train baseline model.
- Fine-tune model.
- Theo dõi loss/accuracy trong quá trình training.
- Đánh giá mô hình trên validation và test set.
- Xuất confusion matrix, classification report và biểu đồ training.
- Cố gắng đạt test accuracy từ 85% trở lên.

Người 3 phải trả lời được:

- MobileNetV2 là gì?
- Vì sao chọn MobileNetV2?
- Transfer learning là gì?
- Fine-tuning khác gì feature extraction?
- Vì sao dùng GlobalAveragePooling?
- Vì sao dùng Dropout?
- Làm sao phát hiện overfitting?
- Mô hình đạt accuracy bao nhiêu?
- Class nào mô hình nhầm nhiều nhất?

---

# 2. Vai trò của MobileNetV2 trong project

MobileNetV2 là mô hình CNN nhẹ, được thiết kế để chạy nhanh và hiệu quả trên thiết bị hạn chế tài nguyên.

Trong project, MobileNetV2 đóng vai trò:

- Một trong ít nhất 2 mô hình bắt buộc.
- Mô hình nhẹ để so sánh với ResNet50.
- Mô hình phù hợp nếu triển khai realtime hoặc trên thiết bị di động.
- Baseline mạnh nhờ pretrained ImageNet.

So sánh vai trò:

| Mô hình | Đặc điểm |
|---|---|
| MobileNetV2 | Nhẹ, nhanh, ít tham số, phù hợp realtime |
| ResNet50 | Nặng hơn, có thể chính xác hơn, phù hợp so sánh |

---

# 3. Kiến thức cần nắm

## 3.1 CNN là gì?

CNN là Convolutional Neural Network, mạng neural tích chập chuyên dùng cho ảnh.

CNN học các đặc trưng theo nhiều mức:

| Tầng | Đặc trưng học được |
|---|---|
| Tầng đầu | Cạnh, đường thẳng, màu sắc |
| Tầng giữa | Hình dạng, góc, texture |
| Tầng sâu | Ký hiệu, đối tượng, class-level feature |

Trong bài toán biển báo:

- Tầng đầu học màu đỏ, xanh, trắng, cạnh tròn/tam giác.
- Tầng giữa học hình dạng biển báo.
- Tầng sâu học ký hiệu STOP, mũi tên, số tốc độ.

## 3.2 MobileNetV2 là gì?

MobileNetV2 là kiến trúc CNN nhẹ do Google đề xuất.

Đặc điểm chính:

- Sử dụng depthwise separable convolution.
- Sử dụng inverted residual block.
- Sử dụng linear bottleneck.
- Giảm số lượng tham số và phép tính.
- Tốc độ inference nhanh.

## 3.3 Depthwise separable convolution

Convolution thông thường xử lý đồng thời không gian và channel, tốn nhiều phép tính.

Depthwise separable convolution tách thành 2 bước:

1. Depthwise convolution: lọc từng channel riêng.
2. Pointwise convolution 1x1: kết hợp thông tin giữa các channel.

Lợi ích:

- Giảm số phép tính.
- Giảm số tham số.
- Tăng tốc inference.

Đây là lý do MobileNetV2 nhẹ hơn nhiều mô hình CNN truyền thống.

## 3.4 Inverted residual block

Trong ResNet truyền thống, block thường:

```text
wide -> narrow -> wide
```

MobileNetV2 dùng inverted residual:

```text
narrow -> wide -> narrow
```

Ý tưởng:

- Mở rộng channel để học đặc trưng.
- Áp dụng depthwise convolution.
- Nén lại bằng linear bottleneck.
- Dùng shortcut connection khi có thể.

## 3.5 Transfer learning

Transfer learning là dùng mô hình đã học từ dataset lớn, ví dụ ImageNet, rồi áp dụng cho bài toán mới.

Lý do dùng:

- Dataset nhóm có hơn 10.000 ảnh, nhưng vẫn nhỏ hơn ImageNet rất nhiều.
- Mô hình pretrained đã học nhiều đặc trưng ảnh cơ bản.
- Giảm thời gian training.
- Tăng accuracy.

Trong project:

```text
MobileNetV2 pretrained ImageNet -> bỏ classification head -> thêm head mới cho class biển báo
```

## 3.6 Feature extraction và fine-tuning

### Feature extraction

- Freeze toàn bộ backbone.
- Chỉ train classification head.
- Nhanh, ít rủi ro overfit.

### Fine-tuning

- Unfreeze một số layer cuối của backbone.
- Train lại với learning rate nhỏ.
- Giúp mô hình thích nghi tốt hơn với biển báo.

Quy trình nên dùng:

```text
Giai đoạn 1: Freeze backbone, train head
Giai đoạn 2: Unfreeze một phần layer cuối, fine-tune
```

---

# 4. Kiến trúc mô hình đề xuất

Mô hình:

```text
Input: 224 x 224 x 3
Backbone: MobileNetV2 pretrained ImageNet, include_top=False
Pooling: GlobalAveragePooling2D
Dropout: 0.3 - 0.5
Dense: num_classes
Activation: Softmax
```

Luồng dữ liệu:

```text
Ảnh biển báo
    ↓
Preprocessing MobileNetV2
    ↓
MobileNetV2 backbone
    ↓
GlobalAveragePooling2D
    ↓
Dropout
    ↓
Dense Softmax
    ↓
Xác suất từng class
```

## 4.1 Vì sao dùng include_top=False?

`include_top=False` nghĩa là bỏ phần classifier gốc của ImageNet.

Lý do:

- ImageNet có 1000 class.
- Project biển báo chỉ có khoảng 8-10 class.
- Cần thay head mới phù hợp với số class của project.

## 4.2 Vì sao dùng GlobalAveragePooling2D?

GlobalAveragePooling2D biến feature map thành vector.

Ưu điểm:

- Giảm số tham số so với Flatten.
- Giảm overfitting.
- Phù hợp transfer learning.

## 4.3 Vì sao dùng Dropout?

Dropout tắt ngẫu nhiên một phần neuron khi training.

Mục đích:

- Giảm overfitting.
- Giúp mô hình không phụ thuộc quá mạnh vào một số đặc trưng.

## 4.4 Vì sao dùng Softmax?

Softmax dùng cho bài toán multi-class classification.

Output:

```text
Xác suất của từng class
```

Class dự đoán:

```text
Class có xác suất cao nhất
```

---

# 5. Quy trình training

## 5.1 Input từ Người 2

Người 3 nhận:

```text
data/split/train/
data/split/val/
data/split/test/
src/features/preprocessing.py
src/features/augmentation.py
```

## 5.2 Giai đoạn 1: Train baseline

Cấu hình:

| Thành phần | Giá trị đề xuất |
|---|---|
| Backbone | Frozen |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 16 hoặc 32 |
| Epochs | 10-15 |
| Loss | Categorical crossentropy |
| Metric | Accuracy |

Mục tiêu:

- Kiểm tra pipeline chạy ổn.
- Có baseline accuracy.
- Xem mô hình có học được không.

## 5.3 Giai đoạn 2: Fine-tuning

Cấu hình:

| Thành phần | Giá trị đề xuất |
|---|---|
| Backbone | Unfreeze một số layer cuối |
| Optimizer | Adam |
| Learning rate | 1e-5 đến 1e-4 |
| Batch size | 16 hoặc 32 |
| Epochs | 10-20 |
| Callback | EarlyStopping, ModelCheckpoint |

Mục tiêu:

- Cải thiện validation accuracy.
- Giúp model học đặc trưng riêng của biển báo.
- Đạt hoặc vượt 85% accuracy trên test set.

## 5.4 Callback cần dùng

### EarlyStopping

Dừng training nếu validation loss không cải thiện.

Lợi ích:

- Tránh overfitting.
- Tiết kiệm thời gian.

### ModelCheckpoint

Lưu model tốt nhất theo validation accuracy hoặc validation loss.

Lợi ích:

- Không mất model tốt nhất.
- Dễ đánh giá lại trên test set.

### ReduceLROnPlateau

Giảm learning rate khi validation loss không cải thiện.

Lợi ích:

- Giúp fine-tuning ổn định hơn.

---

# 6. Chỉ số đánh giá

Người 3 cần xuất:

| Chỉ số | Ý nghĩa |
|---|---|
| Train accuracy | Độ chính xác trên train set |
| Validation accuracy | Độ chính xác trên validation set |
| Test accuracy | Độ chính xác cuối cùng trên test set |
| Train loss | Loss trên train |
| Validation loss | Loss trên validation |
| Precision | Trong các mẫu dự đoán là class X, bao nhiêu mẫu đúng |
| Recall | Trong các mẫu thật sự là class X, bao nhiêu mẫu được tìm thấy |
| F1-score | Trung bình điều hòa precision và recall |
| Confusion matrix | Ma trận thể hiện class đúng/sai |

## 6.1 Accuracy

Accuracy:

```text
Số dự đoán đúng / Tổng số mẫu
```

Phù hợp khi dataset tương đối cân bằng.

## 6.2 Precision

Precision trả lời:

```text
Khi mô hình dự đoán class A, dự đoán đó đúng bao nhiêu phần trăm?
```

## 6.3 Recall

Recall trả lời:

```text
Trong tất cả ảnh thật sự thuộc class A, mô hình tìm đúng bao nhiêu phần trăm?
```

## 6.4 F1-score

F1-score cân bằng precision và recall.

F1 quan trọng khi:

- Class imbalance.
- Cần đánh giá từng class.
- Accuracy tổng thể chưa đủ để kết luận.

## 6.5 Confusion matrix

Confusion matrix giúp biết:

- Class nào dự đoán đúng nhiều.
- Class nào hay bị nhầm.
- Cặp class nào dễ nhầm.

Ví dụ:

```text
turn_left bị nhầm với turn_right
speed_limit bị nhầm với warning
```

---

# 7. Biểu đồ cần tạo

Người 3 cần có:

| Biểu đồ | Mục đích |
|---|---|
| Training accuracy vs validation accuracy | Theo dõi khả năng học |
| Training loss vs validation loss | Phát hiện overfitting |
| Confusion matrix | Phân tích lỗi theo class |
| Per-class F1-score | Xem class nào yếu |
| Prediction sample grid | Minh họa dự đoán đúng/sai |

---

# 8. Dấu hiệu overfitting và underfitting

## 8.1 Overfitting

Dấu hiệu:

```text
Train accuracy cao
Validation accuracy thấp
Train loss giảm
Validation loss tăng
```

Cách xử lý:

- Tăng augmentation.
- Tăng dropout.
- Dùng EarlyStopping.
- Giảm số layer fine-tune.
- Giảm learning rate.
- Bổ sung dữ liệu.

## 8.2 Underfitting

Dấu hiệu:

```text
Train accuracy thấp
Validation accuracy thấp
Loss cả hai đều cao
```

Cách xử lý:

- Train lâu hơn.
- Fine-tune nhiều layer hơn.
- Tăng learning rate hợp lý.
- Kiểm tra preprocessing.
- Kiểm tra label sai.

---

# 9. Benchmark riêng cho MobileNetV2

Người 3 cần hỗ trợ Người 4 đo:

- Model size.
- Số tham số.
- Inference time trung bình mỗi ảnh.
- FPS.

Kỳ vọng:

```text
MobileNetV2 thường nhanh hơn và nhẹ hơn ResNet50.
```

Nếu MobileNetV2 accuracy thấp hơn ResNet50 nhưng chạy nhanh hơn, cần giải thích trade-off:

```text
MobileNetV2 phù hợp khi ưu tiên tốc độ và tài nguyên thấp.
ResNet50 phù hợp khi ưu tiên accuracy.
```

---

# 10. Deliverables bắt buộc

| STT | Deliverable | Mô tả |
|---|---|---|
| 1 | `src/models/train_mobilenetv2.py` | Script train MobileNetV2 |
| 2 | `outputs/models/mobilenetv2_best.*` | Model tốt nhất |
| 3 | `outputs/logs/mobilenetv2_training_log.csv` | Log training |
| 4 | `outputs/metrics/mobilenetv2_metrics.json` | Metrics tổng hợp |
| 5 | `outputs/metrics/mobilenetv2_classification_report.csv` | Precision/recall/F1 |
| 6 | `outputs/figures/mobilenetv2_accuracy.png` | Biểu đồ accuracy |
| 7 | `outputs/figures/mobilenetv2_loss.png` | Biểu đồ loss |
| 8 | `outputs/figures/mobilenetv2_confusion_matrix.png` | Confusion matrix |
| 9 | `outputs/figures/mobilenetv2_predictions.png` | Ảnh dự đoán mẫu |

---

# 11. Tiêu chí hoàn thành

Người 3 hoàn thành khi:

- MobileNetV2 train được end-to-end.
- Có baseline model.
- Có fine-tuned model.
- Có model checkpoint tốt nhất.
- Có biểu đồ loss/accuracy.
- Có classification report.
- Có confusion matrix.
- Có test accuracy.
- Có phân tích lỗi class.
- Có kết quả hỗ trợ benchmark.
- Accuracy đạt từ 85% trở lên hoặc có phân tích nguyên nhân nếu chưa đạt.

---

# 12. Rủi ro và cách xử lý

| Rủi ro | Nguyên nhân | Cách xử lý |
|---|---|---|
| Accuracy thấp | Dữ liệu nhiễu, preprocessing sai | Kiểm tra input, label, data split |
| Overfitting | Model học thuộc train set | Augmentation, dropout, early stopping |
| Validation không tăng | Learning rate chưa phù hợp | Tune learning rate, ReduceLROnPlateau |
| Training lỗi shape | Input không đúng 224x224x3 | Kiểm tra preprocessing |
| Test accuracy thấp hơn validation nhiều | Data leakage hoặc test khó hơn | Kiểm tra split, class distribution |
| Nhầm class có hướng | Augmentation flip sai | Tắt flip hoặc kiểm tra augmentation |

---

# 13. Kiến thức bảo vệ cần nắm

## 13.1 Vì sao chọn MobileNetV2?

Vì MobileNetV2 là mô hình CNN nhẹ, ít tham số, tốc độ inference nhanh và phù hợp với bài toán cần triển khai thực tế như nhận diện biển báo giao thông.

## 13.2 Transfer learning giúp gì?

Transfer learning tận dụng trọng số pretrained từ ImageNet. Các layer đầu đã học được đặc trưng ảnh cơ bản như cạnh, màu sắc và texture, giúp mô hình học nhanh hơn và đạt kết quả tốt hơn khi dataset không quá lớn.

## 13.3 Fine-tuning là gì?

Fine-tuning là mở khóa một số layer cuối của backbone pretrained và huấn luyện tiếp với learning rate nhỏ để mô hình thích nghi tốt hơn với dữ liệu biển báo.

## 13.4 Vì sao dùng learning rate nhỏ khi fine-tune?

Vì trọng số pretrained đã chứa thông tin hữu ích. Learning rate quá lớn có thể làm phá vỡ các đặc trưng đã học, khiến mô hình giảm hiệu quả.

## 13.5 Làm sao biết mô hình bị overfitting?

Nếu train accuracy tiếp tục tăng nhưng validation accuracy không tăng hoặc giảm, đồng thời validation loss tăng, đó là dấu hiệu overfitting.

---

# 14. Câu hỏi bảo vệ mẫu

## Câu 1: MobileNetV2 có ưu điểm gì?

Trả lời gợi ý:

```text
MobileNetV2 có ưu điểm là nhẹ, ít tham số và tốc độ inference nhanh nhờ depthwise separable convolution. Vì vậy mô hình phù hợp với các bài toán cần chạy realtime hoặc trên thiết bị tài nguyên hạn chế.
```

## Câu 2: Nhóm em train MobileNetV2 như thế nào?

Trả lời gợi ý:

```text
Nhóm em sử dụng MobileNetV2 pretrained ImageNet, bỏ classification head gốc, thêm GlobalAveragePooling, Dropout và Dense Softmax theo số class biển báo. Đầu tiên nhóm freeze backbone để train head, sau đó unfreeze một số layer cuối để fine-tune với learning rate nhỏ.
```

## Câu 3: Vì sao không train từ đầu?

Trả lời gợi ý:

```text
Train từ đầu cần dataset rất lớn và thời gian huấn luyện dài. Với project này, transfer learning giúp tận dụng đặc trưng đã học từ ImageNet, giảm thời gian train và tăng khả năng đạt accuracy yêu cầu.
```

## Câu 4: Nếu MobileNetV2 thấp hơn ResNet50 thì có phải mô hình kém không?

Trả lời gợi ý:

```text
Không hẳn. MobileNetV2 được thiết kế ưu tiên tốc độ và kích thước nhỏ. Nếu accuracy thấp hơn ResNet50 nhưng inference nhanh hơn và model nhẹ hơn thì MobileNetV2 vẫn phù hợp trong trường hợp cần realtime hoặc triển khai trên thiết bị hạn chế tài nguyên.
```

## Câu 5: Confusion matrix cho biết điều gì?

Trả lời gợi ý:

```text
Confusion matrix cho biết mô hình dự đoán đúng và sai ở từng class. Nhờ đó nhóm biết class nào dễ nhận diện, class nào hay bị nhầm và có thể phân tích nguyên nhân như hình dạng giống nhau, dữ liệu ít hoặc ảnh nhiễu.
```

---

# 15. Checklist cá nhân Người 3

- [ ] Đã nhận dataset split từ Người 2.
- [ ] Đã kiểm tra preprocessing MobileNetV2.
- [ ] Đã xây dựng model MobileNetV2.
- [ ] Đã train baseline.
- [ ] Đã fine-tune.
- [ ] Đã lưu model tốt nhất.
- [ ] Đã vẽ accuracy curve.
- [ ] Đã vẽ loss curve.
- [ ] Đã xuất classification report.
- [ ] Đã xuất confusion matrix.
- [ ] Đã đo test accuracy.
- [ ] Đã phân tích class nhầm nhiều.
- [ ] Đã hỗ trợ benchmark với Người 4.
- [ ] Đã chuẩn bị giải thích khi bảo vệ.

---

# 16. Kết luận phần Người 3

Người 3 chịu trách nhiệm chứng minh MobileNetV2 là một mô hình phù hợp cho nhận diện biển báo giao thông.

Khi bảo vệ, cần nhấn mạnh:

- MobileNetV2 là mô hình nhẹ và nhanh.
- Transfer learning giúp tận dụng pretrained ImageNet.
- Quy trình train gồm baseline và fine-tuning.
- Kết quả được đánh giá bằng accuracy, precision, recall, F1-score và confusion matrix.
- MobileNetV2 được so sánh với ResNet50 để thấy trade-off giữa tốc độ, kích thước và độ chính xác.
