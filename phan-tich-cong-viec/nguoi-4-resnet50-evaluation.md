# PHÂN TÍCH CÔNG VIỆC NGƯỜI 4
# Vai trò: ResNet50 & Evaluation Engineer

---

# 1. Mục tiêu phần việc

Người 4 chịu trách nhiệm chính cho mô hình **ResNet50** và phần **đánh giá, benchmark, so sánh hai mô hình**.

Mục tiêu:

- Xây dựng mô hình phân loại biển báo giao thông bằng ResNet50.
- Sử dụng transfer learning từ ImageNet.
- Train baseline ResNet50.
- Fine-tune ResNet50.
- Đánh giá ResNet50 trên validation và test set.
- Tổng hợp kết quả MobileNetV2 và ResNet50.
- Benchmark hai mô hình theo cùng điều kiện.
- So sánh accuracy, precision, recall, F1-score, inference time, FPS, model size.
- Kết luận mô hình nào phù hợp hơn theo từng tiêu chí.

Người 4 phải trả lời được:

- ResNet50 là gì?
- Residual learning là gì?
- Vì sao chọn ResNet50?
- ResNet50 khác MobileNetV2 ở đâu?
- Đánh giá mô hình bằng những chỉ số nào?
- Benchmark được đo như thế nào?
- Vì sao accuracy chưa đủ để kết luận?
- Mô hình nào phù hợp realtime?
- Mô hình nào phù hợp nếu ưu tiên accuracy?

---

# 2. Vai trò của ResNet50 trong project

ResNet50 là mô hình CNN sâu, mạnh, được sử dụng rộng rãi trong phân loại ảnh.

Trong project:

- ResNet50 là mô hình thứ hai để đáp ứng yêu cầu ít nhất 2 mô hình.
- Dùng để so sánh với MobileNetV2.
- Đại diện cho mô hình có năng lực biểu diễn mạnh hơn nhưng nặng hơn.
- Giúp nhóm phân tích trade-off giữa độ chính xác và tài nguyên.

So sánh tổng quan:

| Tiêu chí | MobileNetV2 | ResNet50 |
|---|---|---|
| Kích thước | Nhỏ | Lớn hơn |
| Tốc độ | Nhanh | Chậm hơn |
| Số tham số | Ít | Nhiều hơn |
| Khả năng học đặc trưng | Tốt | Rất tốt |
| Phù hợp realtime | Cao | Tùy thiết bị |
| Phù hợp accuracy | Tốt | Thường tốt hơn |

---

# 3. Kiến thức cần nắm

## 3.1 ResNet là gì?

ResNet là Residual Network, một kiến trúc CNN nổi tiếng giúp train mạng rất sâu hiệu quả hơn.

Trước ResNet, khi mạng quá sâu thường gặp vấn đề:

- Gradient khó lan truyền.
- Training khó hội tụ.
- Accuracy có thể giảm khi thêm layer.

ResNet giải quyết bằng residual connection.

## 3.2 Residual learning

Thay vì học trực tiếp hàm:

```text
H(x)
```

ResNet học phần dư:

```text
F(x) = H(x) - x
```

Sau đó output:

```text
H(x) = F(x) + x
```

Ý tưởng:

- Thêm shortcut connection để truyền thông tin.
- Giúp gradient đi qua mạng dễ hơn.
- Cho phép train mạng sâu hơn.

## 3.3 Shortcut connection

Shortcut connection là đường nối bỏ qua một số layer.

Luồng:

```text
x -> convolution layers -> F(x)
 \_______________________/
             ↓
          F(x) + x
```

Lợi ích:

- Giảm vanishing gradient.
- Giữ lại thông tin gốc.
- Giúp mô hình học ổn định hơn.

## 3.4 ResNet50

ResNet50 là phiên bản ResNet có 50 layer.

Đặc điểm:

- Sâu hơn MobileNetV2 theo hướng residual blocks.
- Số tham số lớn hơn.
- Khả năng học đặc trưng mạnh.
- Tốn tài nguyên hơn.

## 3.5 Transfer learning với ResNet50

Tương tự MobileNetV2:

```text
ResNet50 pretrained ImageNet -> bỏ head 1000 class -> thêm head mới cho class biển báo
```

Lý do:

- Dataset project không đủ lớn để train ResNet50 từ đầu.
- Pretrained ImageNet giúp mô hình có đặc trưng ảnh cơ bản.
- Fine-tuning giúp mô hình thích nghi với biển báo.

---

# 4. Kiến trúc mô hình đề xuất

Mô hình:

```text
Input: 224 x 224 x 3
Backbone: ResNet50 pretrained ImageNet, include_top=False
Pooling: GlobalAveragePooling2D
Dropout: 0.4 - 0.5
Dense: num_classes
Activation: Softmax
```

Luồng dữ liệu:

```text
Ảnh biển báo
    ↓
Preprocessing ResNet50
    ↓
ResNet50 backbone
    ↓
GlobalAveragePooling2D
    ↓
Dropout
    ↓
Dense Softmax
    ↓
Xác suất từng class
```

## 4.1 Vì sao ResNet50 có thể overfit?

ResNet50 có nhiều tham số hơn MobileNetV2.

Nếu dataset không đủ đa dạng:

- Mô hình có thể học thuộc train set.
- Train accuracy cao nhưng validation/test thấp.

Cách giảm overfitting:

- Augmentation hợp lý.
- Dropout.
- EarlyStopping.
- Fine-tune ít layer hơn.
- Learning rate nhỏ.

## 4.2 Vì sao dùng dropout cao hơn MobileNetV2?

Vì ResNet50 thường có năng lực học mạnh hơn và nhiều tham số hơn. Dropout 0.4-0.5 có thể giúp giảm overfitting ở classification head.

---

# 5. Quy trình training ResNet50

## 5.1 Input từ Người 2

Người 4 nhận:

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

- Kiểm tra ResNet50 chạy ổn với dataset.
- Có baseline để so sánh sau fine-tuning.

## 5.3 Giai đoạn 2: Fine-tuning

Cấu hình:

| Thành phần | Giá trị đề xuất |
|---|---|
| Backbone | Unfreeze block cuối |
| Optimizer | Adam hoặc SGD momentum |
| Learning rate | 1e-5 đến 1e-4 |
| Batch size | 16 hoặc 32 |
| Epochs | 10-20 |
| Callback | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |

Lưu ý:

- Không unfreeze quá nhiều layer ngay từ đầu.
- Learning rate cần nhỏ để không phá pretrained weights.
- Theo dõi validation loss để tránh overfitting.

---

# 6. Evaluation tổng hợp

Người 4 không chỉ đánh giá ResNet50 mà còn tổng hợp kết quả của cả MobileNetV2.

## 6.1 Các chỉ số bắt buộc

| Chỉ số | Ý nghĩa |
|---|---|
| Accuracy | Tỉ lệ dự đoán đúng |
| Precision | Độ chính xác trong các dự đoán dương |
| Recall | Khả năng tìm đúng mẫu thuộc class |
| F1-score | Cân bằng precision và recall |
| Confusion matrix | Phân tích lỗi theo class |
| Inference time | Thời gian dự đoán |
| FPS | Số ảnh xử lý mỗi giây |
| Model size | Dung lượng model |
| Parameters | Số tham số |

## 6.2 Macro, micro, weighted average

### Macro average

Tính trung bình chỉ số trên từng class, mỗi class có trọng số như nhau.

Phù hợp khi muốn xem mô hình có công bằng giữa các class không.

### Weighted average

Tính trung bình theo số lượng mẫu mỗi class.

Phù hợp khi dataset có class imbalance.

### Micro average

Tính trên tổng tất cả dự đoán.

Với multi-class single-label, micro F1 thường gần accuracy.

## 6.3 Vì sao accuracy chưa đủ?

Accuracy có thể cao nhưng mô hình vẫn kém ở class ít dữ liệu.

Ví dụ:

| Class | Số ảnh | Recall |
|---|---:|---:|
| stop | 2000 | 95% |
| speed_limit | 2000 | 94% |
| yield | 300 | 50% |

Accuracy tổng có thể vẫn cao, nhưng class `yield` bị nhận diện kém.

Vì vậy cần thêm:

- Precision.
- Recall.
- F1-score.
- Confusion matrix.

---

# 7. Benchmark

## 7.1 Mục tiêu benchmark

Benchmark dùng để so sánh hiệu quả thực tế của hai mô hình.

Không chỉ hỏi:

```text
Mô hình nào chính xác hơn?
```

Mà còn hỏi:

```text
Mô hình nào nhanh hơn?
Mô hình nào nhẹ hơn?
Mô hình nào phù hợp triển khai realtime?
```

## 7.2 Điều kiện benchmark công bằng

Hai mô hình phải được đo trong cùng điều kiện:

- Cùng test set.
- Cùng image size.
- Cùng preprocessing tương ứng.
- Cùng thiết bị.
- Cùng batch size nếu đo batch inference.
- Cùng số lần warm-up nếu đo thời gian.

## 7.3 Cách đo inference time

Quy trình đề xuất:

```text
Load model
Warm-up vài batch
Chạy dự đoán trên test set
Đo tổng thời gian
Tính thời gian trung bình mỗi ảnh
Tính FPS
```

Công thức:

```text
inference_time_per_image = total_time / number_of_images
FPS = number_of_images / total_time
```

## 7.4 Model size

Model size là dung lượng file model lưu trên ổ đĩa.

Ý nghĩa:

- Model nhỏ dễ triển khai.
- Model lớn tốn bộ nhớ.
- Model nhỏ thường tải nhanh hơn.

## 7.5 Number of parameters

Số tham số thể hiện độ lớn mô hình.

Thông thường:

- Nhiều tham số hơn có thể học tốt hơn.
- Nhưng cũng dễ overfit hơn.
- Và inference chậm hơn.

---

# 8. Bảng so sánh bắt buộc

Người 4 cần tạo bảng:

| Metric | MobileNetV2 | ResNet50 | Nhận xét |
|---|---:|---:|---|
| Test accuracy | | | |
| Macro precision | | | |
| Macro recall | | | |
| Macro F1-score | | | |
| Weighted F1-score | | | |
| Inference time/image | | | |
| FPS | | | |
| Model size | | | |
| Parameters | | | |

Nhận xét cần rõ:

- Mô hình nào chính xác hơn.
- Mô hình nào nhanh hơn.
- Mô hình nào nhẹ hơn.
- Sự đánh đổi giữa accuracy và tốc độ.

---

# 9. Biểu đồ cần tạo

| Biểu đồ | Mục đích |
|---|---|
| ResNet50 accuracy curve | Theo dõi training |
| ResNet50 loss curve | Phát hiện overfitting |
| ResNet50 confusion matrix | Phân tích lỗi |
| Bar chart accuracy 2 model | So sánh độ chính xác |
| Bar chart F1-score 2 model | So sánh cân bằng precision/recall |
| Bar chart inference time | So sánh tốc độ |
| Bar chart model size | So sánh dung lượng |
| Per-class F1 comparison | Xem model nào tốt hơn theo class |

---

# 10. Kết luận kỹ thuật cần viết

Người 4 cần đưa ra kết luận theo nhiều góc nhìn.

## 10.1 Nếu ResNet50 accuracy cao hơn

Có thể kết luận:

```text
ResNet50 có khả năng học đặc trưng mạnh hơn, giúp đạt accuracy cao hơn. Tuy nhiên mô hình có kích thước lớn và inference chậm hơn MobileNetV2.
```

## 10.2 Nếu MobileNetV2 nhanh hơn

Có thể kết luận:

```text
MobileNetV2 phù hợp hơn khi cần triển khai realtime hoặc trên thiết bị tài nguyên hạn chế do model nhẹ và tốc độ inference nhanh.
```

## 10.3 Nếu hai mô hình accuracy gần nhau

Có thể kết luận:

```text
Khi accuracy hai mô hình gần nhau, MobileNetV2 có lợi thế hơn về tốc độ và kích thước. ResNet50 chỉ nên chọn nếu cần độ ổn định hoặc F1-score tốt hơn ở một số class quan trọng.
```

## 10.4 Nếu accuracy dưới 85%

Cần phân tích:

- Dữ liệu có nhiễu không?
- Class có bị imbalance không?
- Augmentation có phù hợp không?
- Learning rate có quá cao/thấp không?
- Fine-tuning đã đủ chưa?
- Class nào bị nhầm nhiều?

Không nên chỉ nói:

```text
Mô hình chưa tốt.
```

Phải có nguyên nhân kỹ thuật và hướng cải thiện.

---

# 11. Deliverables bắt buộc

| STT | Deliverable | Mô tả |
|---|---|---|
| 1 | `src/models/train_resnet50.py` | Script train ResNet50 |
| 2 | `src/evaluation/evaluate_model.py` | Script đánh giá model |
| 3 | `src/evaluation/benchmark.py` | Script benchmark |
| 4 | `src/evaluation/compare_models.py` | Script so sánh model |
| 5 | `outputs/models/resnet50_best.*` | Model ResNet50 tốt nhất |
| 6 | `outputs/metrics/resnet50_metrics.json` | Metrics ResNet50 |
| 7 | `outputs/metrics/model_comparison.csv` | Bảng so sánh 2 model |
| 8 | `outputs/figures/resnet50_accuracy.png` | Accuracy curve |
| 9 | `outputs/figures/resnet50_loss.png` | Loss curve |
| 10 | `outputs/figures/resnet50_confusion_matrix.png` | Confusion matrix |
| 11 | `outputs/figures/model_accuracy_comparison.png` | So sánh accuracy |
| 12 | `outputs/figures/model_speed_comparison.png` | So sánh tốc độ |
| 13 | `outputs/figures/model_size_comparison.png` | So sánh model size |

---

# 12. Tiêu chí hoàn thành

Người 4 hoàn thành khi:

- ResNet50 train được end-to-end.
- Có baseline và fine-tuned model.
- Có model checkpoint tốt nhất.
- Có test accuracy.
- Có classification report.
- Có confusion matrix.
- Có benchmark MobileNetV2 và ResNet50.
- Có bảng so sánh hai mô hình.
- Có biểu đồ so sánh.
- Có kết luận kỹ thuật rõ ràng.
- Accuracy đạt yêu cầu 85% hoặc có phân tích nguyên nhân và phương án cải thiện.

---

# 13. Rủi ro và cách xử lý

| Rủi ro | Nguyên nhân | Cách xử lý |
|---|---|---|
| ResNet50 overfit | Mô hình lớn, dữ liệu chưa đủ đa dạng | Dropout, augmentation, EarlyStopping |
| Training chậm | ResNet50 nặng | Giảm batch size, dùng GPU, giảm epoch thử nghiệm |
| Out of memory | Batch size quá lớn | Giảm batch size xuống 16 hoặc 8 |
| Accuracy thấp | Fine-tuning chưa tốt | Tune learning rate, unfreeze block cuối |
| Benchmark không công bằng | Khác test set hoặc thiết bị | Chuẩn hóa điều kiện đo |
| Kết luận thiếu thuyết phục | Chỉ nhìn accuracy | Thêm F1, inference time, model size |

---

# 14. Kiến thức bảo vệ cần nắm

## 14.1 Vì sao chọn ResNet50?

ResNet50 là kiến trúc CNN mạnh, phổ biến trong phân loại ảnh, có residual connection giúp train mạng sâu hiệu quả. Dùng ResNet50 giúp nhóm có mô hình mạnh để so sánh với MobileNetV2.

## 14.2 Residual connection giải quyết vấn đề gì?

Residual connection giúp thông tin và gradient truyền qua mạng sâu dễ hơn, giảm vấn đề vanishing gradient và giúp mô hình học ổn định khi số layer lớn.

## 14.3 Vì sao ResNet50 có thể chính xác hơn nhưng chậm hơn?

ResNet50 có nhiều layer và nhiều tham số hơn nên khả năng biểu diễn đặc trưng mạnh hơn. Tuy nhiên điều này cũng làm tăng thời gian tính toán, dung lượng model và tài nguyên cần dùng.

## 14.4 Vì sao cần benchmark?

Benchmark giúp đánh giá mô hình không chỉ bằng accuracy mà còn bằng tốc độ, kích thước và khả năng triển khai. Trong thực tế, một mô hình chính xác hơn nhưng quá chậm có thể không phù hợp realtime.

## 14.5 Vì sao cần F1-score?

F1-score cân bằng precision và recall, đặc biệt hữu ích khi dữ liệu có class imbalance hoặc khi cần đánh giá hiệu quả từng class.

---

# 15. Câu hỏi bảo vệ mẫu

## Câu 1: ResNet50 khác MobileNetV2 ở điểm nào?

Trả lời gợi ý:

```text
ResNet50 là mô hình CNN sâu dùng residual connection, có khả năng học đặc trưng mạnh nhưng nặng hơn. MobileNetV2 dùng depthwise separable convolution nên nhẹ và nhanh hơn, phù hợp realtime hơn.
```

## Câu 2: Vì sao cần so sánh hai mô hình?

Trả lời gợi ý:

```text
Đề bài yêu cầu ít nhất 2 mô hình. Việc so sánh giúp nhóm đánh giá trade-off giữa độ chính xác, tốc độ inference và kích thước model, từ đó chọn mô hình phù hợp với từng mục tiêu triển khai.
```

## Câu 3: Nếu ResNet50 accuracy cao hơn MobileNetV2 thì chọn ResNet50 luôn được không?

Trả lời gợi ý:

```text
Không nhất thiết. Nếu mục tiêu là accuracy cao nhất thì ResNet50 có thể phù hợp. Nhưng nếu cần realtime hoặc triển khai trên thiết bị tài nguyên thấp thì MobileNetV2 có thể tốt hơn vì nhẹ và nhanh hơn.
```

## Câu 4: Benchmark inference time được đo như thế nào?

Trả lời gợi ý:

```text
Nhóm em load model, chạy warm-up vài batch, sau đó dự đoán trên cùng test set và đo tổng thời gian. Từ đó tính inference time trung bình mỗi ảnh và FPS. Hai mô hình được đo cùng thiết bị, cùng test set và cùng image size.
```

## Câu 5: Accuracy đạt 85% có đủ để kết luận model tốt không?

Trả lời gợi ý:

```text
Accuracy 85% đáp ứng yêu cầu tối thiểu, nhưng chưa đủ để kết luận toàn diện. Nhóm còn xem precision, recall, F1-score, confusion matrix, inference time và model size để đánh giá đầy đủ hơn.
```

---

# 16. Checklist cá nhân Người 4

- [ ] Đã nhận dataset split từ Người 2.
- [ ] Đã kiểm tra preprocessing ResNet50.
- [ ] Đã xây dựng model ResNet50.
- [ ] Đã train baseline.
- [ ] Đã fine-tune.
- [ ] Đã lưu model tốt nhất.
- [ ] Đã vẽ accuracy curve.
- [ ] Đã vẽ loss curve.
- [ ] Đã xuất classification report ResNet50.
- [ ] Đã xuất confusion matrix ResNet50.
- [ ] Đã nhận metrics MobileNetV2 từ Người 3.
- [ ] Đã benchmark cả hai mô hình.
- [ ] Đã tạo bảng so sánh.
- [ ] Đã tạo biểu đồ so sánh.
- [ ] Đã viết kết luận kỹ thuật.
- [ ] Đã chuẩn bị giải thích khi bảo vệ.

---

# 17. Kết luận phần Người 4

Người 4 chịu trách nhiệm chứng minh ResNet50 là mô hình so sánh mạnh và đưa ra đánh giá tổng hợp cho toàn project.

Khi bảo vệ, Người 4 cần nhấn mạnh:

- ResNet50 dùng residual learning để train mạng sâu hiệu quả.
- Quy trình train gồm baseline và fine-tuning.
- Đánh giá mô hình không chỉ dựa vào accuracy.
- Benchmark giúp so sánh khả năng triển khai thực tế.
- Kết luận cần dựa trên nhiều tiêu chí: accuracy, F1-score, tốc độ, kích thước và tài nguyên.
