# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.
1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về cả bức ảnh.
- Vì sao: Tác vụ phân loại ảnh (Image Classification) dự đoán nhãn cho toàn bộ khung hình tổng thể, gán ý nghĩa cho toàn ảnh chứ không định vị hay khoanh vùng tọa độ vật thể riêng lẻ nào.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Do đi kèm theo taxonomy/checkpoint của mô hình đã được huấn luyện sẵn trên tập dữ liệu chuẩn (như ImageNet-1K), hoàn toàn không phải do mô hình tự nghĩ ra.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- class_id giúp hệ thống máy tính xử lý và truy vấn chính xác, tránh lỗi định dạng chuỗi.
- class_name giúp con người đọc hiểu trực quan.
- taxonomy_name xác định rõ hệ thống phân loại (bộ nhãn chuẩn) đang được áp dụng, tránh nhầm lẫn giữa các tập dữ liệu khác nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Cần quy định rõ ưu tiên gán nhãn cho chủ đề hoặc vật thể chiếm diện tích lớn nhất, nằm ở vị trí trung tâm, hoặc là chủ đề chính phục vụ mục đích khai thác của dự án.
6. Vì sao không được lấy `score` của model làm ground truth?
- score chỉ phản ánh mức độ tự tin (xác suất thống kê) của mô hình đối với dự đoán đó, không đảm bảo tính chính xác thực tế. Mô hình hoàn toàn có thể tự tin dự đoán sai, trong khi ground truth đòi hỏi tiêu chuẩn chính xác do con người quy định theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= person ; `score`= 0.912625; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`= 113.58 ; `bbox_height`= 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Góc trên-trái: Nằm tại tọa độ pixel $(385.33, 69.24)$ (cách mép trái 385.33 pixel và cách mép trên 69.24 pixel).
- Góc dưới-phải: Nằm tại tọa độ pixel $(498.92, 348.92)$ (cách mép trái 498.92 pixel và cách mép trên 348.92 pixel).
- Kích thước: Hộp giới hạn có chiều rộng là 113.58 pixel và chiều cao là 279.68 pixel bao trọn lấy đối tượng.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- Ngưỡng 0.20 (Ngưỡng thấp): Số lượng prediction sẽ nhiều hơn vì mô hình giữ lại tất cả các hộp có độ tự tin từ 20% trở lên (chấp nhận cả các dự đoán có xác suất thấp, dễ xuất hiện dương tính giả).
- Ngưỡng 0.60 (Ngưỡng cao): Số lượng prediction sẽ ít hơn vì mô hình chỉ lọc giữ lại các hộp có độ tự tin từ 60% trở lên (chỉ hiển thị những đối tượng mà mô hình cực kỳ chắc chắn).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Khi hạ ngưỡng, số lượng hộp dự đoán tăng lên kèm theo nhiều hộp rác hoặc nhận diện nhầm. Reviewer phải thực hiện thêm việc rà soát kỹ lưỡng, thủ công từng hộp có độ tin cậy thấp để loại bỏ các dương tính giả (false positives) và chỉnh sửa lại các hộp bị lệch biên.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Quy tắc Bounding Box: "Mọi hộp giới hạn (bounding box) phải ôm sát các biên ngoài cùng của phần hiển thị thực tế của vật thể. Không được để khoảng trống quá lớn (padding) xung quanh vật thể và không được cắt cụt các phần chính thuộc về đối tượng cần gán nhãn."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Vật bị cắt mép / che khuất: Chiếc lò nướng (oven 0.69) ở phía bên trái bị cắt mép bởi khung hình biên của ảnh
- Cách xử lý: Đưa vào guideline thay vì escalation.
- Vì sao: Việc vật thể bị che khuất một phần hoặc bị cắt mép khung hình là một hiện tượng phổ biến và lặp lại trong các bài toán gán nhãn dữ liệu thực tế. Guideline chuẩn cần phải định nghĩa sẵn ranh giới rõ ràng (ví dụ: chỉ gán nhãn nếu phần thân vật thể lộ diện trên 50% hoặc vẫn vẽ hộp bao trọn phần nhìn thấy được tới mép ảnh), giúp annotator chủ động thực thi mà không cần mất thời gian hỏi lại qua escalation.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-011 ; `class_name`= spoon ; `score`= 0.359147 ; số điểm polygon=`polygon_point_count`= 37 ; 3–5 điểm đầu của `polygon_xy`= [[529.0, 66.0], [529.0, 72.0], [528.0, 73.0], [528.0, 74.0], [527.0, 75.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- bounding box chỉ tạo ra một hình chữ nhật thô bao quanh bên ngoài, thì polygon cung cấp đường viền đa giác ôm sát từng đường cong, góc cạnh và hình thù thực tế của đối tượng. Điều này giúp xác định chính xác pixel nào thuộc về vật thể và pixel nào thuộc về nền (background).
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- Ý nghĩa: instance_id dùng để phân biệt từng thể hiện (instance) riêng biệt của cùng một loại đối tượng xuất hiện trong một khung hình cụ thể
- Không phải class ID vì nó không định nghĩa loại vật thể (ví dụ: thìa, bát, người), mà chỉ là mã số định danh phân biệt các đối tượng độc lập.
- Không phải tracking ID vì đây là bài toán ảnh tĩnh (single image), nó không theo dõi và duy trì định danh của đối tượng xuyên suốt qua các khung hình video theo thời gian.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Quy tắc vẽ Polygon Mask: "Đường viền đa giác (polygon) phải bám sát ranh giới hình học thực tế của vật thể. Tại các đoạn uốn lượn hoặc bo tròn, annotator phải bấm điểm dày đặc để ôm khít đường cong, tuyệt đối không để phần mask bị lẹm vào nền hoặc tràn lan sang các vật thể lân cận."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Chỗ cần chỉ ra: Khu vực các chiếc bát (bowl với các nhãn 0.70, 0.55, 0.74) đặt sát và chạm nhau / chồng lên nhau ở góc dưới bên trái bàn
- Cách xử lý: Sử dụng guideline trước, nếu trường hợp quá mơ hồ mới dùng escalation.
- Vì sao: Các tình huống như vật thể bị che khuất một phần, ranh giới tiếp giáp bị mờ hoặc hai vật dụng chạm sát vào nhau là những hiện tượng thường gặp trong gán nhãn dữ liệu. Guideline chuẩn cần quy định rõ nguyên tắc xử lý cụ thể (ví dụ: tách biên tại vị trí giao nhau theo nếp gấp thực tế của vật). Annotator phải ưu tiên tuân thủ guideline; chỉ khi ranh giới hoàn toàn không thể phân biệt bằng mắt thường mới cần đưa lên escalation hỏi Lab Coach.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn đơn lẻ (ImageNet-1K) | Ảnh nhiều chủ đề, dễ nhận diện nhầm lớp phụ | Gán đúng một lớp chủ đạo theo guideline | Kiểm tra lớp chọn có đúng chủ thể nổi bật không |
| Phát hiện vật thể | Hộp bbox_xyxy + tên lớp | Hộp bị lệch, vật bị che khuất một phần | Khoanh bounding box ôm sát biên vật thể | Kiểm tra sót vật thể hoặc thừa/thiếu biên hộp |
| Instance segmentation | Đa giác chi tiết polygon_xy | Biên mờ, các vật chạm nhau khó tách điểm | Kẻ chuỗi điểm polygon bám sát biên hình học | Kiểm tra độ mượt polygon, không lẹm nền |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Tuyệt đối không tải lên hoặc sử dụng bất kỳ hình ảnh cá nhân, thông tin nhạy cảm hay dữ liệu nội bộ nào lên môi trường Colab hay repository công khai.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, em sẽ dừng thao tác và báo cáo ngay cho giảng viên hoặc Lab Coach phụ trách lớp học.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
