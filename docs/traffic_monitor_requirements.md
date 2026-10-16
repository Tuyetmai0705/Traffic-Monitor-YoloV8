# PROJECT PROMPT — BÀI 9: TRAFFIC MONITOR DESKTOP APPLICATION

## 1. Vai trò

Bạn là Senior Python Computer Vision Engineer và Software Architect.

Hãy triển khai hoàn chỉnh dự án:

**Traffic Monitor — Intelligent Traffic Monitoring Desktop Application**

Dự án được xây dựng dựa trên giáo trình Supervision, trong đó Bài 9 là sản phẩm cuối kết hợp các chức năng:

- YOLO object detection
- vehicle classification
- Supervision `sv.Detections`
- ByteTrack multi-object tracking
- LineZone vehicle counting
- ViewTransformer perspective transformation
- vehicle speed estimation
- speed violation detection
- realtime statistics
- annotated video export
- JSON report export

Không được viết toàn bộ hệ thống vào một file duy nhất.

---

# 2. Công nghệ bắt buộc

Python 3.10+

Computer Vision:
- OpenCV
- Ultralytics YOLO
- Supervision

Desktop UI:
- PySide6

Data:
- NumPy
- JSON
- dataclasses

Testing:
- pytest

---

# 3. Kiến trúc hệ thống

Sử dụng kiến trúc module rõ ràng:

```text
traffic_monitor/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── app/
│   ├── config/
│   │   ├── settings.py
│   │   └── default_config.json
│   │
│   ├── core/
│   │   ├── detector.py
│   │   ├── tracker.py
│   │   ├── counter.py
│   │   ├── speed_estimator.py
│   │   ├── violation_detector.py
│   │   └── traffic_monitor.py
│   │
│   ├── models/
│   │   ├── detection_result.py
│   │   ├── traffic_statistics.py
│   │   └── violation.py
│   │
│   ├── pipeline/
│   │   ├── frame_processor.py
│   │   └── video_processor.py
│   │
│   ├── services/
│   │   ├── report_service.py
│   │   ├── video_service.py
│   │   └── config_service.py
│   │
│   ├── ui/
│   │   ├── main_window.py
│   │   ├── video_widget.py
│   │   ├── dashboard_widget.py
│   │   ├── control_panel.py
│   │   ├── settings_dialog.py
│   │   └── styles.py
│   │
│   ├── utils/
│   │   ├── display.py
│   │   ├── logger.py
│   │   ├── file_utils.py
│   │   └── time_utils.py
│   │
│   └── constants/
│       ├── classes.py
│       └── messages.py
│
├── models/
│   └── yolov8n.pt
│
├── data/
│   ├── input/
│   └── calibration/
│
├── outputs/
│   ├── videos/
│   ├── reports/
│   └── logs/
│
└── tests/
```

---

# 4. Vehicle Detection

Tạo:

`app/core/detector.py`

Class:

```python
VehicleDetector
```

Các hàm bắt buộc:

```python
__init__()
load_model()
detect()
filter_detections()
get_class_name()
get_statistics()
```

Pipeline:

```text
frame
↓
YOLO inference
↓
sv.Detections.from_ultralytics()
↓
confidence filtering
↓
vehicle class filtering
↓
return detections
```

Vehicle classes:

```python
CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7
```

Default confidence:

```text
0.30
```

---

# 5. Tracking

Tạo:

`app/core/tracker.py`

Class:

```python
VehicleTracker
```

Sử dụng:

```python
sv.ByteTrack
```

Các hàm:

```python
__init__()
update()
reset()
get_tracker_ids()
```

Sau tracking phải có:

```python
detections.tracker_id
```

Sử dụng smoothing để giảm rung:

```python
sv.DetectionsSmoother
```

---

# 6. Vehicle Counting

Tạo:

`app/core/counter.py`

Class:

```python
VehicleCounter
```

Sử dụng:

```python
sv.LineZone
```

Các chức năng:

- total IN
- total OUT
- car IN
- car OUT
- motorcycle IN
- motorcycle OUT
- bus IN
- bus OUT
- truck IN
- truck OUT

Các hàm:

```python
count()
get_total_in()
get_total_out()
get_class_counts()
reset()
```

Chỉ gọi `line_zone.trigger()` một lần cho mỗi frame.

Anchor:

```python
sv.Position.BOTTOM_CENTER
```

---

# 7. Speed Estimation

Tạo:

`app/core/speed_estimator.py`

Class:

```python
SpeedEstimator
```

Sử dụng perspective transformation.

Input:

```text
pixel coordinates
```

Output:

```text
real-world coordinates
```

Sau đó tính:

```text
distance / time × 3.6 = km/h
```

Lưu lịch sử tọa độ theo:

```python
tracker_id
```

Sử dụng:

```python
deque
```

Không tính tốc độ chỉ từ hai frame liên tiếp.

Các hàm:

```python
transform_points()
update_track()
calculate_speed()
get_speed()
reset()
```

---

# 8. Speed Violation

Tạo:

`app/core/violation_detector.py`

Class:

```python
SpeedViolationDetector
```

Default:

```text
speed_limit = 80 km/h
```

Các hàm:

```python
check()
is_violation()
register_violation()
get_violations()
reset()
```

Nếu:

```text
speed > speed_limit
```

thì đánh dấu vi phạm.

Lưu:

```text
tracker_id
maximum detected speed
```

---

# 9. Traffic Monitor

Tạo:

`app/core/traffic_monitor.py`

Class:

```python
TrafficMonitor
```

Class này chỉ orchestration.

Không viết toàn bộ thuật toán vào đây.

Nó phải sử dụng:

```python
VehicleDetector
VehicleTracker
VehicleCounter
SpeedEstimator
SpeedViolationDetector
```

Các hàm:

```python
process_frame()
get_statistics()
reset()
```

---

# 10. Frame Pipeline

Tạo:

`app/pipeline/frame_processor.py`

Pipeline bắt buộc:

```text
Frame
 ↓
Detection
 ↓
Filtering
 ↓
Tracking
 ↓
Counting
 ↓
Speed Estimation
 ↓
Violation Detection
 ↓
Annotation
 ↓
Statistics
 ↓
Processed Frame
```

Không để UI thực hiện các bước Computer Vision.

---

# 11. Video Processing

Tạo:

`app/pipeline/video_processor.py`

Class:

```python
VideoProcessor
```

Các hàm:

```python
open()
read_frame()
process()
pause()
resume()
stop()
release()
```

Phải hỗ trợ:

```text
Open video
Start
Pause
Resume
Stop
```

Không đọc toàn bộ video vào RAM.

Xử lý frame tuần tự.

---

# 12. Desktop Application

Sử dụng PySide6.

Tạo:

```text
main_window.py
video_widget.py
dashboard_widget.py
control_panel.py
settings_dialog.py
```

Main Window:

```text
┌─────────────────────────────────────────────────────────────┐
│                    TRAFFIC MONITOR                         │
├─────────────────────────────────┬───────────────────────────┤
│                                 │ TRAFFIC STATISTICS        │
│                                 │                           │
│          VIDEO                 │ Total IN                  │
│          PREVIEW               │ Total OUT                 │
│                                 │ Car                       │
│                                 │ Motorcycle                │
│                                 │ Bus                       │
│                                 │ Truck                     │
│                                 │ Violations                │
│                                 │ Average Speed             │
├─────────────────────────────────┴───────────────────────────┤
│ Open Video | Start | Pause | Resume | Stop | Settings      │
├─────────────────────────────────────────────────────────────┤
│ Status | FPS | Processing Time                              │
└─────────────────────────────────────────────────────────────┘
```

---

# 13. UI Responsibilities

## MainWindow

Chịu trách nhiệm:

- tạo layout
- kết nối signals
- điều phối UI

Không chứa YOLO code.

## VideoWidget

Chịu trách nhiệm:

- hiển thị frame
- resize frame
- clear frame

## DashboardWidget

Hiển thị:

```text
Total IN
Total OUT

Car IN/OUT
Motorcycle IN/OUT
Bus IN/OUT
Truck IN/OUT

Violations
Average Speed
Maximum Speed
```

## ControlPanel

Có:

```text
Open Video
Start
Pause
Resume
Stop
Export Report
Settings
```

---

# 14. Settings

Cho phép người dùng cấu hình:

```text
YOLO model
Confidence threshold
Speed limit
Vehicle classes
Output video path
Report path
Perspective source points
Road width
Road length
```

Không hard-code các thông số này trong core logic.

---

# 15. Annotation

Frame kết quả phải hiển thị:

- bounding box
- vehicle class
- confidence
- tracker ID
- trace
- speed km/h
- speed violation
- counting line
- IN/OUT count
- realtime dashboard

---

# 16. Realtime Dashboard

Dashboard phải cập nhật sau mỗi frame hoặc theo interval phù hợp.

Ví dụ:

```text
TOTAL IN: 47
TOTAL OUT: 52

CAR IN: 40
CAR OUT: 48

TRUCK IN: 5
TRUCK OUT: 3

BUS IN: 2
BUS OUT: 1

MOTORCYCLE IN: 0
MOTORCYCLE OUT: 0

SPEED VIOLATIONS: 2
```

---

# 17. Output

Khi Stop hoặc video kết thúc:

Tự động xuất:

```text
outputs/
├── videos/
│   └── traffic_result.mp4
│
└── reports/
    └── traffic_report.json
```

JSON phải chứa:

```json
{
    "video": "...",
    "total_in": 0,
    "total_out": 0,
    "by_class": {},
    "speed_violations": {}
}
```

---

# 18. Threading

Không được chạy YOLO inference trực tiếp trên UI thread.

Sử dụng:

```text
QThread
```

hoặc worker thread.

Kiến trúc:

```text
Main UI Thread
      │
      │ signals
      ▼
Video Processing Worker
      │
      ├── YOLO
      ├── ByteTrack
      ├── LineZone
      ├── Speed
      └── Annotation
      │
      ▼
Processed Frame
      │
      ▼
Main UI Thread
```

UI phải luôn responsive trong lúc xử lý video.

---

# 19. State Management

Application phải có các trạng thái:

```text
IDLE
VIDEO_SELECTED
RUNNING
PAUSED
STOPPED
COMPLETED
ERROR
```

Không cho phép:

```text
Start khi chưa chọn video
Pause khi chưa chạy
Resume khi chưa pause
Export khi chưa có result
```

Hiển thị message phù hợp cho người dùng.

---

# 20. Error Handling

Phải xử lý:

- video không tồn tại
- model không tồn tại
- video không mở được
- model load lỗi
- video codec lỗi
- invalid configuration
- invalid perspective points
- output directory không tồn tại
- YOLO inference error

Không để application crash với traceback cho người dùng cuối.

---

# 21. Logging

Tạo:

```text
outputs/logs/app.log
```

Log:

```text
Application started
Video loaded
Model loaded
Processing started
Processing paused
Processing resumed
Processing stopped
Report exported
Processing completed
Error
```

---

# 22. Testing

Tạo unit tests:

```text
tests/
├── test_detector.py
├── test_tracker.py
├── test_counter.py
├── test_speed.py
└── test_report.py
```

Test ít nhất:

- detector initialization
- filtering
- counting
- speed calculation
- violation detection
- report generation

---

# 23. Coding Rules

Bắt buộc:

1. Không viết tất cả code vào một file.
2. Không viết YOLO inference trong UI.
3. Không viết business logic trong widget.
4. Không hard-code configuration.
5. Mỗi class có một trách nhiệm chính.
6. Function phải ngắn và có mục đích rõ ràng.
7. Type hints.
8. Docstring cho public functions.
9. Exception handling.
10. Logging.
11. Không duplicate code.
12. Không sử dụng global state nếu không cần thiết.
13. Không đọc toàn bộ video vào RAM.
14. UI không được freeze khi inference.
15. Tách Core / Pipeline / UI / Services / Utils.

---

# 24. Development Order

Không được viết ngẫu nhiên.

Triển khai theo thứ tự:

### Phase 1
Project structure

### Phase 2
Configuration

### Phase 3
VehicleDetector

### Phase 4
VehicleTracker

### Phase 5
VehicleCounter

### Phase 6
SpeedEstimator

### Phase 7
ViolationDetector

### Phase 8
TrafficMonitor

### Phase 9
FrameProcessor

### Phase 10
VideoProcessor

### Phase 11
ReportService

### Phase 12
Desktop UI

### Phase 13
QThread integration

### Phase 14
Settings

### Phase 15
Error handling

### Phase 16
Testing

### Phase 17
README

---

# 25. Cách trả code

Khi triển khai project:

Không trả pseudo-code.

Không dùng:

```python
...
```

hoặc:

```python
# implement here
```

Mỗi file phải là code hoàn chỉnh.

Triển khai từng file theo thứ tự.

Mỗi lần trả một file phải ghi:

```text
FILE:
app/core/detector.py
```

Sau đó đưa toàn bộ code của file.

Giải thích ngắn:

```text
Mục đích
Các function
Input
Output
Dependency
```

Sau khi hoàn thành tất cả file:

1. kiểm tra import
2. kiểm tra circular import
3. kiểm tra đường dẫn
4. kiểm tra requirements
5. kiểm tra flow Start/Pause/Resume/Stop
6. kiểm tra report
7. kiểm tra UI
8. kiểm tra toàn bộ project có thể chạy bằng:

```bash
python main.py
```

---

# 26. Tiêu chí hoàn thành

Project chỉ được xem là hoàn thành khi:

```text
python main.py
```

mở được Desktop App.

User có thể:

1. chọn video
2. chọn model
3. chỉnh confidence
4. chỉnh speed limit
5. Start
6. xem video realtime
7. thấy bounding boxes
8. thấy tracker IDs
9. thấy trace
10. thấy tốc độ
11. thấy IN/OUT
12. thấy thống kê từng loại xe
13. thấy cảnh báo quá tốc độ
14. Pause
15. Resume
16. Stop
17. xuất annotated video
18. xuất JSON report

Không được bỏ qua bất kỳ chức năng nào.