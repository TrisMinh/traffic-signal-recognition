# PHÂN TÍCH CÔNG VIỆC NGƯỜI 1
# Vai trò: Data Collection Engineer

---

# 1. Mục tiêu phần việc

Người 1 chịu trách nhiệm chính cho phần **thu thập dữ liệu** của project nhận diện biển báo giao thông.

Đây là phần nền tảng của toàn bộ đồ án. Nếu dữ liệu thu thập không đủ số lượng, không rõ nguồn, nhiều ảnh sai nhãn hoặc chất lượng thấp thì các bước phía sau như trích xuất đặc trưng, huấn luyện MobileNetV2, huấn luyện ResNet50 và đánh giá mô hình đều bị ảnh hưởng.

Mục tiêu bắt buộc:

- Tự crawl dữ liệu ảnh biển báo giao thông.
- Dataset sau làm sạch phải còn hơn 10.000 ảnh hợp lệ.
- Có dẫn nguồn dữ liệu rõ ràng.
- Có mô tả cách thức thu thập dữ liệu.
- Có metadata chứng minh quá trình crawl.
- Có thống kê mô tả và trực quan hóa dữ liệu ban đầu.

Kết quả của Người 1 phải trả lời được các câu hỏi:

- Dữ liệu được lấy từ đâu?
- Vì sao chọn các nguồn đó?
- Crawl bằng cách nào?
- Có bao nhiêu class?
- Mỗi class có bao nhiêu ảnh?
- Làm sao chứng minh đây là dữ liệu tự crawl?
- Dữ liệu có đủ lớn và đủ cân bằng không?
- Những rủi ro chất lượng dữ liệu là gì?

---

# 2. Vai trò của dữ liệu trong bài toán nhận diện biển báo

Bài toán nhận diện biển báo giao thông là bài toán phân loại ảnh.

Input:

```text
Ảnh biển báo giao thông
```

Output:

```text
Nhãn class của biển báo
```

Ví dụ:

| Ảnh đầu vào | Nhãn đầu ra |
|---|---|
| Ảnh biển báo STOP | `stop` |
| Ảnh biển báo cấm vào | `no_entry` |
| Ảnh biển báo giới hạn tốc độ | `speed_limit` |
| Ảnh biển báo người đi bộ | `pedestrian_crossing` |

Trong học sâu, mô hình không học từ định nghĩa bằng chữ mà học từ dữ liệu. Vì vậy dataset cần:

- Đủ nhiều mẫu.
- Đủ đa dạng về góc chụp, ánh sáng, nền ảnh, chất lượng ảnh.
- Đúng nhãn.
- Ít ảnh trùng lặp.
- Ít ảnh nhiễu hoặc không liên quan.
- Phân bố class không quá lệch.

Nếu dữ liệu kém, mô hình có thể:

- Học sai đặc trưng.
- Overfit vào ảnh trùng lặp.
- Nhận diện tốt trên validation nhưng kém trên ảnh mới.
- Nhầm giữa các class giống nhau.
- Không đạt accuracy 85%.

---

# 3. Kiến thức cần nắm

## 3.1 Web crawling là gì?

Web crawling là quá trình tự động thu thập dữ liệu từ Internet.

Trong project này, crawling nghĩa là:

- Tạo danh sách từ khóa tìm kiếm ảnh.
- Truy cập nguồn ảnh.
- Lấy URL ảnh hoặc trang chứa ảnh.
- Tải ảnh về máy.
- Lưu thông tin nguồn vào metadata.
- Tổ chức ảnh theo từng class.

Ví dụ:

```text
Query: "stop sign road"
Source: Google Images
Class: stop
Output: data/raw/stop/stop_google_000001.jpg
```

## 3.2 Tự crawl khác gì dùng dataset có sẵn?

Tự crawl:

- Nhóm tự viết hoặc sử dụng script để tải ảnh từ các nguồn công khai.
- Có lưu URL nguồn và ngày crawl.
- Có thể chứng minh quá trình thu thập.
- Phù hợp yêu cầu đề bài.

Dataset có sẵn:

- Tải nguyên bộ dữ liệu từ Kaggle, Roboflow, GitHub hoặc các kho dataset.
- Không thể hiện rõ công sức tự thu thập.
- Có thể không được tính nếu đề yêu cầu tự crawl.

Vì đề yêu cầu **SV phải tự thu thập dữ liệu**, Người 1 không nên xem Kaggle là nguồn dữ liệu chính.

Kaggle chỉ nên dùng để:

- Tham khảo tên class.
- Tham khảo cấu trúc dataset.
- Đối chiếu chất lượng ảnh.
- Không tính vào mốc hơn 10.000 ảnh tự crawl.

## 3.3 Metadata là gì?

Metadata là dữ liệu mô tả dữ liệu.

Với mỗi ảnh, metadata cần ghi:

- Ảnh thuộc class nào.
- Ảnh được crawl từ nguồn nào.
- URL ảnh hoặc URL trang chứa ảnh.
- Từ khóa dùng để tìm ảnh.
- Ngày crawl.
- Kích thước ảnh.
- Trạng thái ảnh sau kiểm tra.

Metadata giúp:

- Chứng minh dữ liệu có nguồn gốc.
- Kiểm tra lại ảnh nếu bị sai nhãn.
- Thống kê số ảnh theo nguồn.
- Thống kê số ảnh theo class.
- Hỗ trợ báo cáo và bảo vệ.

## 3.4 Class imbalance là gì?

Class imbalance là hiện tượng số lượng mẫu giữa các class bị lệch nhiều.

Ví dụ:

| Class | Số ảnh |
|---|---:|
| stop | 2.500 |
| no_entry | 2.300 |
| speed_limit | 2.000 |
| yield | 400 |

Ở ví dụ trên, class `yield` quá ít. Mô hình có thể học kém class này và thường dự đoán sai.

Người 1 cần kiểm soát số lượng ảnh thô mỗi class ngay từ lúc crawl.

Nguyên tắc:

```text
Không chỉ cần tổng ảnh >10.000, mà mỗi class cũng phải đủ dữ liệu.
```

## 3.5 Duplicate image là gì?

Duplicate image là ảnh trùng hoặc gần trùng.

Ví dụ:

- Cùng một ảnh được tải từ nhiều nguồn.
- Cùng một ảnh nhưng khác kích thước.
- Cùng một ảnh nhưng bị nén lại.

Ảnh trùng gây vấn đề:

- Dataset nhìn có vẻ lớn nhưng thực tế ít đa dạng.
- Mô hình dễ overfit.
- Nếu ảnh trùng xuất hiện cả train và test, accuracy bị ảo.

Người 1 không cần xử lý duplicate chính, nhưng cần hỗ trợ Người 2 bằng cách lưu metadata và hạn chế crawl lặp URL.

---

# 4. Danh sách class đề xuất

Để vừa đủ yêu cầu hơn 10.000 ảnh, vừa khả thi trong 3 tuần, nên chọn từ 8 đến 10 class.

Danh sách class đề xuất:

| STT | Tên class trong code | Tên tiếng Việt | Lý do chọn |
|---|---|---|---|
| 1 | `stop` | Biển báo dừng | Phổ biến, dễ nhận diện |
| 2 | `no_entry` | Biển cấm vào | Hình dạng đặc trưng |
| 3 | `speed_limit` | Biển giới hạn tốc độ | Quan trọng trong giao thông |
| 4 | `pedestrian_crossing` | Biển người đi bộ | Phổ biến |
| 5 | `traffic_light` | Đèn giao thông | Nhiều ảnh thực tế |
| 6 | `yield` | Biển nhường đường | Dễ phân biệt |
| 7 | `no_parking` | Biển cấm đỗ xe | Phổ biến |
| 8 | `warning` | Biển cảnh báo | Nhiều biến thể |
| 9 | `turn_left` | Biển rẽ trái | Có thể bổ sung |
| 10 | `turn_right` | Biển rẽ phải | Có thể bổ sung |

Khuyến nghị:

- Nếu sợ thiếu thời gian, chọn 8 class.
- Nếu crawl tốt và dữ liệu cân bằng, chọn 10 class.
- Không nên chọn quá nhiều class nếu không đủ ảnh cho mỗi class.

Mục tiêu số lượng:

```text
Ảnh thô: 13.000 - 14.000 ảnh
Ảnh sạch sau lọc: > 10.000 ảnh
```

Lý do cần crawl thô nhiều hơn 10.000:

- Một phần ảnh sẽ bị lỗi.
- Một phần ảnh sẽ trùng.
- Một phần ảnh bị mờ.
- Một phần ảnh sai nội dung.
- Một phần ảnh kích thước quá nhỏ.

---

# 5. Nguồn dữ liệu

## 5.1 Nguồn nên dùng

| Nguồn | Ưu điểm | Lưu ý |
|---|---|---|
| Google Images | Nhiều ảnh, đa dạng | Cần lưu query và URL |
| Bing Images | Nhiều ảnh, dễ bổ sung | Cần lọc nhiễu |
| Wikimedia Commons | Ảnh có nguồn rõ | Số lượng có thể ít |
| Flickr | Ảnh thực tế đa dạng | Cần kiểm tra license nếu báo cáo |
| Open Images | Có nhãn và URL | Cần lọc đúng class |
| Website giao thông | Ảnh đúng chủ đề | Có thể ít mẫu |

## 5.2 Nguồn nên tránh dùng làm nguồn chính

| Nguồn | Lý do |
|---|---|
| Kaggle dataset tải sẵn | Không thể hiện rõ tự crawl |
| Dataset GitHub tải sẵn | Dễ bị xem là dùng lại dữ liệu |
| Roboflow public dataset | Có thể không phù hợp yêu cầu tự thu thập |

Nếu dùng các nguồn này để tham khảo, cần ghi rõ:

```text
Không tính vào dataset chính dùng để đáp ứng yêu cầu tự crawl.
```

---

# 6. Từ khóa crawl

Mỗi class nên có nhiều query.

Ví dụ:

| Class | Query tiếng Anh | Query tiếng Việt |
|---|---|---|
| `stop` | stop sign, road stop sign, traffic stop sign | biển báo dừng |
| `no_entry` | no entry sign, do not enter sign | biển cấm vào |
| `speed_limit` | speed limit sign, road speed limit sign | biển giới hạn tốc độ |
| `pedestrian_crossing` | pedestrian crossing sign, crosswalk sign | biển người đi bộ |
| `traffic_light` | traffic light, road traffic signal | đèn giao thông |
| `yield` | yield sign, give way sign | biển nhường đường |
| `no_parking` | no parking sign, parking prohibited sign | biển cấm đỗ xe |
| `warning` | warning traffic sign, road warning sign | biển cảnh báo giao thông |
| `turn_left` | turn left sign, left turn road sign | biển rẽ trái |
| `turn_right` | turn right sign, right turn road sign | biển rẽ phải |

Lưu ý khi chọn query:

- Query quá rộng sẽ nhiều ảnh nhiễu.
- Query quá hẹp sẽ ít ảnh.
- Nên dùng cả tiếng Anh và tiếng Việt.
- Có thể thêm từ khóa `road`, `traffic`, `sign`, `street` để giảm nhiễu.

---

# 7. Quy trình thực hiện

## 7.1 Bước 1: Chốt class

Người 1 họp nhanh với nhóm để chốt:

- Số class.
- Tên class trong code.
- Tên tiếng Việt.
- Số ảnh mục tiêu mỗi class.
- Quy tắc loại ảnh sai nội dung.

Kết quả cần có:

```text
data/metadata/classes.csv
```

Nội dung gợi ý:

| class_name | vietnamese_name | target_raw_count | target_clean_count |
|---|---|---:|---:|
| stop | Biển báo dừng | 1300 | 1000 |
| no_entry | Biển cấm vào | 1300 | 1000 |

## 7.2 Bước 2: Tạo config crawl

File đề xuất:

```text
src/crawl/crawl_config.yaml
```

Nội dung cần có:

- Class name.
- Danh sách query.
- Nguồn crawl.
- Số ảnh mục tiêu.
- Thư mục lưu.

Ví dụ:

```yaml
classes:
  stop:
    queries:
      - "stop sign"
      - "road stop sign"
      - "traffic stop sign"
    target_count: 1300
  no_entry:
    queries:
      - "no entry sign"
      - "do not enter sign"
    target_count: 1300
```

## 7.3 Bước 3: Crawl ảnh

Script crawl cần:

- Đọc config.
- Duyệt từng class.
- Duyệt từng query.
- Lấy URL ảnh.
- Tải ảnh về.
- Lưu theo đúng class.
- Ghi metadata.
- Ghi log lỗi.

Thư mục lưu ảnh thô:

```text
data/raw/<class_name>/
```

Ví dụ:

```text
data/raw/stop/stop_google_000001.jpg
data/raw/no_entry/no_entry_bing_000001.jpg
```

## 7.4 Bước 4: Kiểm tra số lượng ảnh thô

Sau khi crawl, cần thống kê:

- Tổng số ảnh.
- Số ảnh mỗi class.
- Số ảnh từ từng nguồn.
- Class nào thiếu ảnh.

Nếu class thiếu ảnh:

- Bổ sung query.
- Dùng nguồn khác.
- Crawl thêm tiếng Việt hoặc tiếng Anh.

## 7.5 Bước 5: Bàn giao cho Người 2

Người 1 bàn giao:

- Dataset raw.
- Metadata.
- Config crawl.
- Danh sách class.
- Thống kê ảnh thô.

Người 2 sẽ tiếp tục:

- Loại ảnh lỗi.
- Loại ảnh trùng.
- Loại ảnh mờ.
- Chia train/validation/test.

---

# 8. Metadata chi tiết

File metadata chính:

```text
data/metadata/images_metadata.csv
```

Cột bắt buộc:

| Cột | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `image_id` | string | ID duy nhất của ảnh |
| `class_name` | string | Nhãn class |
| `source_site` | string | Tên website nguồn |
| `source_url` | string | URL ảnh hoặc trang chứa ảnh |
| `query` | string | Từ khóa dùng để crawl |
| `crawl_date` | string | Ngày crawl |
| `raw_path` | string | Đường dẫn ảnh thô |
| `width` | int | Chiều rộng ảnh nếu đọc được |
| `height` | int | Chiều cao ảnh nếu đọc được |
| `download_status` | string | success hoặc failed |
| `error_message` | string | Lỗi nếu có |

Cột có thể bổ sung sau cleaning:

| Cột | Ý nghĩa |
|---|---|
| `clean_path` | Đường dẫn ảnh sạch |
| `is_duplicate` | Có bị trùng không |
| `is_corrupted` | Có bị lỗi không |
| `is_blurry` | Có bị mờ không |
| `final_status` | valid hoặc removed |

Metadata tốt giúp khi bảo vệ có thể nói:

```text
Nhóm em tự crawl dữ liệu, mỗi ảnh đều có lưu URL nguồn, query crawl, ngày crawl và class tương ứng. Sau đó nhóm kiểm tra chất lượng và loại các ảnh lỗi/trùng/mờ trước khi huấn luyện.
```

---

# 9. Thống kê và trực quan hóa cần làm

Người 1 phụ trách thống kê dữ liệu thô và nguồn crawl.

Biểu đồ cần có:

## 9.1 Số ảnh theo class

Mục đích:

- Kiểm tra class imbalance.
- Chứng minh dataset đủ lớn.

Dạng biểu đồ:

```text
Bar chart
```

Trục X:

```text
Class
```

Trục Y:

```text
Số ảnh
```

## 9.2 Số ảnh theo nguồn

Mục đích:

- Cho thấy dữ liệu lấy từ nhiều nguồn.
- Tránh phụ thuộc vào một nguồn duy nhất.

Dạng biểu đồ:

```text
Bar chart hoặc pie chart
```

## 9.3 Số ảnh theo query

Mục đích:

- Biết query nào hiệu quả.
- Biết query nào gây nhiễu hoặc ít ảnh.

## 9.4 Sample images theo class

Mục đích:

- Minh họa trực quan dữ liệu.
- Hỗ trợ bảo vệ.

Mỗi class nên có:

```text
8 - 16 ảnh mẫu
```

---

# 10. Deliverables bắt buộc

Người 1 cần bàn giao:

| STT | Deliverable | Mô tả |
|---|---|---|
| 1 | `src/crawl/crawl_images.py` | Script crawl ảnh |
| 2 | `src/crawl/crawl_config.yaml` | Config class/query/số lượng |
| 3 | `data/raw/` | Ảnh thô theo class |
| 4 | `data/metadata/classes.csv` | Danh sách class |
| 5 | `data/metadata/images_metadata.csv` | Metadata ảnh |
| 6 | `outputs/figures/raw_class_distribution.png` | Biểu đồ số ảnh theo class |
| 7 | `outputs/figures/source_distribution.png` | Biểu đồ số ảnh theo nguồn |
| 8 | `outputs/figures/raw_sample_grid.png` | Ảnh mẫu từng class |
| 9 | `outputs/metrics/raw_data_summary.csv` | Bảng thống kê dữ liệu thô |

---

# 11. Tiêu chí hoàn thành

Người 1 hoàn thành khi:

- Có danh sách class rõ ràng.
- Có script crawl hoặc quy trình crawl có thể chạy lại.
- Có hơn 13.000 ảnh thô, hoặc đủ để sau clean còn hơn 10.000 ảnh.
- Có metadata nguồn cho ảnh.
- Có thống kê số lượng ảnh theo class.
- Có thống kê số lượng ảnh theo nguồn.
- Có ảnh mẫu minh họa từng class.
- Bàn giao được dataset raw cho Người 2.

---

# 12. Rủi ro và cách xử lý

| Rủi ro | Nguyên nhân | Cách xử lý |
|---|---|---|
| Không đủ ảnh | Query hẹp, nguồn ít | Thêm query, thêm nguồn |
| Ảnh sai class nhiều | Query quá rộng | Thêm từ khóa cụ thể, lọc thủ công một phần |
| Ảnh trùng nhiều | Crawl nhiều nguồn giống nhau | Lưu URL, hỗ trợ duplicate detection |
| Ảnh chất lượng thấp | Ảnh nhỏ, mờ, nén mạnh | Crawl dư số lượng để Người 2 lọc |
| Class imbalance | Class khó tìm ảnh | Crawl bổ sung class thiếu |
| Không chứng minh tự crawl | Không lưu metadata | Bắt buộc lưu source_url/query/crawl_date |

---

# 13. Kiến thức bảo vệ cần nắm

Người 1 cần giải thích được:

## 13.1 Vì sao cần tự crawl?

Vì đề bài yêu cầu sinh viên tự thu thập dữ liệu. Tự crawl giúp nhóm chủ động xây dựng dataset, kiểm soát class, nguồn ảnh, số lượng ảnh và có thể mô tả quy trình thu thập dữ liệu rõ ràng.

## 13.2 Vì sao phải crawl nhiều hơn 10.000 ảnh thô?

Vì sau quá trình làm sạch sẽ loại bỏ ảnh lỗi, ảnh trùng, ảnh mờ, ảnh sai nội dung hoặc ảnh quá nhỏ. Nếu chỉ crawl đúng 10.000 ảnh thì dataset sạch có thể không đạt yêu cầu.

## 13.3 Vì sao cần metadata?

Metadata giúp chứng minh nguồn gốc dữ liệu, hỗ trợ thống kê, hỗ trợ kiểm tra lại ảnh sai và giúp quy trình thu thập dữ liệu có tính minh bạch.

## 13.4 Vì sao không dùng Kaggle làm nguồn chính?

Vì Kaggle thường cung cấp dataset đã được người khác thu thập sẵn. Đề bài yêu cầu tự crawl nên dataset có sẵn không thể hiện đúng yêu cầu thu thập dữ liệu của nhóm.

## 13.5 Vì sao cần dữ liệu cân bằng?

Nếu một class có quá nhiều ảnh và class khác có quá ít ảnh, mô hình có xu hướng học tốt class nhiều ảnh và kém class ít ảnh. Điều này làm giảm recall/F1-score ở các class thiểu số.

---

# 14. Câu hỏi bảo vệ mẫu

## Câu 1: Nhóm em thu thập dữ liệu từ đâu?

Trả lời gợi ý:

```text
Nhóm em tự crawl ảnh từ nhiều nguồn công khai như Google Images, Bing Images, Wikimedia Commons, Flickr và một số website liên quan đến biển báo giao thông. Với mỗi ảnh, nhóm lưu lại class, URL nguồn, website nguồn, query crawl và ngày crawl trong file metadata.
```

## Câu 2: Làm sao chứng minh dataset là tự crawl?

Trả lời gợi ý:

```text
Nhóm em có script crawl, file config query, thư mục ảnh thô và file metadata. Metadata lưu source_url, source_site, query và crawl_date cho từng ảnh. Đây là bằng chứng cho quá trình tự thu thập dữ liệu.
```

## Câu 3: Vì sao dataset raw cần lớn hơn dataset clean?

Trả lời gợi ý:

```text
Trong quá trình làm sạch, nhóm phải loại ảnh lỗi, ảnh trùng, ảnh mờ, ảnh sai nội dung và ảnh quá nhỏ. Vì vậy nhóm crawl dư khoảng 13.000 đến 14.000 ảnh thô để đảm bảo sau làm sạch vẫn còn hơn 10.000 ảnh hợp lệ.
```

## Câu 4: Nếu một class có ít ảnh hơn các class khác thì xử lý thế nào?

Trả lời gợi ý:

```text
Nhóm em kiểm tra thống kê số lượng ảnh theo class. Nếu class nào thiếu, nhóm bổ sung query, dùng thêm nguồn crawl hoặc mở rộng từ khóa bằng tiếng Anh và tiếng Việt để tăng số ảnh cho class đó.
```

## Câu 5: Vì sao cần thống kê dữ liệu trước khi train?

Trả lời gợi ý:

```text
Thống kê giúp nhóm biết dataset có đủ số lượng không, class có bị lệch không, nguồn dữ liệu có đa dạng không và dữ liệu có cần bổ sung không. Đây là bước quan trọng trước khi preprocessing và huấn luyện mô hình.
```

---

# 15. Checklist cá nhân Người 1

Trước khi bàn giao, Người 1 tự kiểm tra:

- [ ] Đã chốt danh sách class.
- [ ] Đã có query cho từng class.
- [ ] Đã crawl đủ ảnh thô.
- [ ] Đã lưu ảnh đúng thư mục class.
- [ ] Đã có metadata cho ảnh.
- [ ] Đã có thống kê số ảnh theo class.
- [ ] Đã có thống kê số ảnh theo nguồn.
- [ ] Đã có sample images từng class.
- [ ] Đã bàn giao dataset raw cho Người 2.
- [ ] Đã chuẩn bị phần giải thích khi bảo vệ.

---

# 16. Kết luận phần Người 1

Người 1 là người chịu trách nhiệm tạo nền dữ liệu cho toàn bộ project.

Nếu phần này làm tốt, nhóm sẽ có:

- Dataset đủ lớn.
- Dataset có nguồn rõ ràng.
- Bằng chứng tự crawl.
- Phân bố class hợp lý.
- Nền tảng tốt để preprocessing, feature engineering và training model.

Khi bảo vệ, Người 1 cần nhấn mạnh:

- Nhóm tự crawl dữ liệu.
- Dataset sạch sau lọc lớn hơn 10.000 ảnh.
- Có metadata nguồn đầy đủ.
- Có thống kê và trực quan hóa dữ liệu.
- Dữ liệu được tổ chức hợp lý để phục vụ huấn luyện mô hình.
