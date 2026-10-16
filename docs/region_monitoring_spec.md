# Tài liệu kỹ thuật: Giám sát đối tượng trong vùng chỉ định

**Tên tiếng Anh:** Region-Based Object Monitoring  
**Phiên bản:** 1.0

## 1. Mục tiêu

Xây dựng ứng dụng computer vision phân tích video/camera để xác định đối tượng trong một hoặc nhiều vùng quan tâm (*Region of Interest – ROI*), theo dõi từng đối tượng qua thời gian, đếm số đối tượng đang hiện diện và tính thời gian lưu trú (*dwell time*).

Hệ thống cần trả lời các câu hỏi:

- Hiện tại vùng A có bao nhiêu đối tượng?
- Đối tượng ID nào đang ở trong vùng?
- Mỗi đối tượng đã ở trong vùng bao lâu?
- Đối tượng vào/ra vùng lúc nào?
- Đối tượng nào vượt thời gian lưu trú cho phép?

## 2. Phạm vi và giả định

- Đầu vào MVP: file video; có thể mở rộng sang camera stream.
- Đối tượng được cấu hình theo lớp của model: `person`, `car`, `motorcycle`, `bus`, `truck` hoặc lớp tùy chỉnh.
- ROI là polygon do người dùng cấu hình trên frame.
- Mỗi đối tượng được gán `track_id`. ID có thể thay đổi khi đối tượng bị che khuất lâu hoặc tracker mất dấu.
- Video file nên dùng media timestamp/FPS để tính dwell time; camera live dùng đồng hồ thời gian thực.
- `Occupancy` (số hiện tại trong vùng) khác với tổng lượt vào/ra.

## 3. Kiến trúc đề xuất

```text
Video Source
    ↓
Frame Reader
    ↓
Object Detector
    ↓
Multi-Object Tracker
    ↓
ROI Membership Evaluator
    ↓
Dwell-Time Manager
    ↓
Event / Statistics Service
    ↓
Annotation + Desktop Dashboard
    ↓
JSON / CSV Report
```

### 3.1. Trách nhiệm các module

| Module | Trách nhiệm | Đầu ra chính |
|---|---|---|
| `VideoProcessor` | Đọc frame tuần tự, lấy timestamp/FPS, pause/resume/stop | Frame, timestamp |
| `ObjectDetector` | Phát hiện và lọc theo confidence/class | Bounding box, class, confidence |
| `ObjectTracker` | Duy trì ID qua các frame | `track_id` + detection |
| `ROIManager` | Lưu polygon, xác định anchor trong/ngoài vùng | Membership theo ROI |
| `DwellTimeTracker` | Quản lý thời điểm vào, thời gian lưu trú và ra | Session/event |
| `OccupancyCounter` | Đếm track hiện diện theo ROI ở frame hiện tại | Occupancy |
| `EventService` | Ghi sự kiện `ENTER`/`EXIT` và cảnh báo | Event records |
| `ReportService` | Xuất báo cáo JSON/CSV | Report file |
| `Desktop UI` | Chọn video, cấu hình ROI, xem video/dashboard | UI state |

## 4. Thuật toán cốt lõi

### 4.1. Detection và tracking

Mỗi frame đi qua detector; các detection được chuyển sang ByteTrack hoặc tracker tương đương để duy trì `track_id`. Detection đơn lẻ không đủ để tính dwell time ổn định vì hệ thống cần nhận biết cùng một đối tượng qua nhiều frame.

### 4.2. Xác định đối tượng trong ROI

ROI là polygon theo tọa độ pixel của frame gốc. Với mỗi tracked object, chọn anchor ổn định (ví dụ `bottom-center` cho người/xe) hoặc centroid tùy nghiệp vụ, sau đó kiểm tra điểm đó nằm trong polygon. Phải ánh xạ tọa độ đúng nếu video preview đã resize.

> Kiểm tra anchor nằm trong polygon không giống kiểm tra toàn bộ bounding box giao với polygon. Cần chọn một quy tắc và giữ nhất quán.

### 4.3. Occupancy — số đối tượng hiện có

Tại frame \(t\), occupancy của ROI bằng số `track_id` duy nhất được xác định đang ở trong ROI. Occupancy là số hiện tại, không phải tổng lượt vào tích lũy.

### 4.4. Dwell time — thời gian lưu trú

1. Khi track chuyển từ ngoài vào trong ROI, lưu `entry_time`.
2. Khi còn ở trong vùng, tính `dwell_time = current_time - entry_time`.
3. Khi track ra khỏi vùng, lưu `exit_time` và `final_dwell_time = exit_time - entry_time`.

Nên có `grace period` hoặc `timeout` để tránh trạng thái trong/ngoài chập chờn ở biên hoặc mất track vài frame. Khi đối tượng ra rồi quay lại, bản MVP nên tạo một visit/session mới; nếu cần tổng thời gian theo ca thì bổ sung metric riêng.

### 4.5. Công thức

- **Dwell time phiên:** `T_dwell = T_exit - T_entry`
- **Occupancy:** `Occupancy(ROI, t) = số track_id duy nhất inside ROI tại thời điểm t`
- **Entry count:** số sự kiện `ENTER` hợp lệ; không dùng occupancy thay thế metric này.

## 5. Mô hình dữ liệu

### 5.1. ROI configuration

```json
{
  "roi_id": "zone_a",
  "name": "Khu vực A",
  "polygon": [[100, 100], [400, 100], [450, 350], [80, 350]],
  "allowed_classes": ["person"],
  "max_dwell_seconds": 300
}
```

### 5.2. Track session

```json
{
  "track_id": 18,
  "class_name": "person",
  "roi_id": "zone_a",
  "entry_time": 12.4,
  "last_seen_time": 18.7,
  "exit_time": null,
  "dwell_seconds": 6.3,
  "status": "inside"
}
```

### 5.3. Event

```json
{
  "event_id": "event-001",
  "track_id": 18,
  "roi_id": "zone_a",
  "event_type": "ENTER",
  "timestamp": 12.4,
  "class_name": "person"
}
```

## 6. Thiết kế UI đề xuất

- **Video Preview:** bounding boxes, `track_id`, ROI polygon, trạng thái IN/OUT và dwell time.
- **ROI Editor:** thêm/sửa/xóa vùng, đặt tên, chọn lớp đối tượng và ngưỡng thời gian lưu trú.
- **Dashboard:** occupancy hiện tại theo ROI; tổng lượt vào/ra; số cảnh báo dwell time.
- **Object Table:** `track_id`, class, ROI, entry time, dwell time hiện tại, trạng thái.
- **Controls:** Open Video/Camera, Start, Pause, Resume, Stop, Export Report, Settings.
- **Event Log:** thời gian vào/ra, vượt ngưỡng, mất track và lỗi nguồn video.

## 7. Cấu trúc thư mục gợi ý

```text
region_monitor/
├── main.py
├── requirements.txt
├── README.md
├── app/
│   ├── config/
│   │   ├── settings.py
│   │   └── default_config.json
│   ├── core/
│   │   ├── detector.py
│   │   ├── tracker.py
│   │   ├── roi_manager.py
│   │   ├── occupancy_counter.py
│   │   └── dwell_time_tracker.py
│   ├── models/
│   │   ├── detection.py
│   │   ├── roi.py
│   │   ├── track_session.py
│   │   └── event.py
│   ├── pipeline/
│   │   ├── frame_processor.py
│   │   └── video_processor.py
│   ├── services/
│   │   ├── event_service.py
│   │   └── report_service.py
│   ├── ui/
│   │   ├── main_window.py
│   │   ├── video_widget.py
│   │   ├── roi_editor.py
│   │   ├── dashboard_widget.py
│   │   ├── object_table.py
│   │   └── settings_dialog.py
│   └── utils/
│       ├── logger.py
│       └── geometry.py
├── data/
│   └── videos/
├── outputs/
│   ├── reports/
│   └── logs/
└── tests/
    ├── test_roi_membership.py
    ├── test_occupancy.py
    ├── test_dwell_time.py
    └── test_report.py
```

## 8. Trường hợp biên cần xử lý

- **Đối tượng chạm biên ROI:** quy định rõ phép kiểm tra inside và anchor được dùng.
- **ID thay đổi do che khuất:** không mặc định hai ID khác nhau là cùng một đối tượng.
- **Track mất vài frame:** dùng lost-track timeout; thời gian có gián đoạn có thể chỉ là ước lượng.
- **Video pause:** thời gian theo media-time phải dừng cùng video; camera live dùng thời gian thực.
- **Đối tượng rời rồi quay lại:** tạo session mới; có thể tính tổng dwell theo track riêng.
- **ROI được vẽ trên preview:** quy đổi về tọa độ frame gốc trước khi xử lý.
- **ROI chồng lấn:** quy định một track có thể thuộc nhiều ROI hay chỉ một ROI.

## 9. Tiêu chí nghiệm thu MVP

- [ ] Mở video và vẽ/lưu được ít nhất một polygon ROI.
- [ ] Đối tượng có `track_id` hiển thị ổn định trong các tình huống thông thường.
- [ ] Dashboard thể hiện occupancy hiện tại theo từng ROI.
- [ ] Mỗi track trong ROI có `entry_time` và dwell time cập nhật.
- [ ] Khi track rời ROI, hệ thống ghi exit event và thời gian phiên.
- [ ] Có cảnh báo khi dwell time vượt ngưỡng cấu hình.
- [ ] Xuất danh sách session và thống kê JSON/CSV.
- [ ] Có unit tests cho ROI membership, occupancy, dwell time và timeout.

## 10. Công nghệ đề xuất

- Python 3.10+
- OpenCV: đọc/ghi video.
- Ultralytics YOLO: object detection.
- ByteTrack hoặc tracker tương đương: duy trì ID.
- Supervision `PolygonZone` hoặc phép kiểm tra point-in-polygon: ROI.
- PySide6: desktop UI.
- `dataclasses`: mô hình dữ liệu.
- `pytest`: unit tests.

Cần kiểm tra và khóa phiên bản thư viện trước khi triển khai. Một số thư viện cung cấp sẵn region counting, nhưng dwell time theo từng track vẫn cần logic session riêng để lưu thời điểm vào/ra.

## 11. Tài liệu tham khảo

1. **Ultralytics — Region Counting:** https://docs.ultralytics.com/guides/region-counting/  
   Đếm occupancy hiện tại trong polygon và phân biệt với đếm lượt đi qua.
2. **Ultralytics — TrackZone:** https://docs.ultralytics.com/guides/trackzone/  
   Theo dõi đối tượng trong vùng xác định và thống kê số track trong vùng.
3. **Supervision — PolygonZone:** https://supervision.roboflow.com/latest/detection/tools/polygon_zone/  
   Polygon ROI, triggering anchors và sự phụ thuộc vào `tracker_id`.

## 12. Phân biệt các metric

| Metric | Câu hỏi | Ví dụ |
|---|---|---|
| Occupancy | Hiện tại có bao nhiêu đối tượng? | 7 người đang ở Zone A |
| Entry count | Có bao nhiêu lượt vào? | 23 lượt vào trong video |
| Dwell time | Một đối tượng đã ở bao lâu? | Track 18 ở 42,5 giây |
| Overstay | Ai vượt ngưỡng lưu trú? | Track 18 vượt 300 giây |
