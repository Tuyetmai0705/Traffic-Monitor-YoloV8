# 🚦 Traffic Monitor: giám sát giao thông thông minh với YOLOv8

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![YOLOv8n](https://img.shields.io/badge/YOLOv8n-Ultralytics-111F68)
![Supervision](https://img.shields.io/badge/Supervision-ByteTrack%20%7C%20LineZone-7E56C2)
![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%20GPU-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![PySide6](https://img.shields.io/badge/UI-PySide6%20(Qt6)-41CD52?logo=qt&logoColor=white)
![Tests](https://img.shields.io/badge/tests-102%20passed-2EA44F)

**Traffic Monitor** là ứng dụng desktop phân tích video giao thông trên GPU. Ứng dụng phát hiện và phân loại phương tiện bằng **YOLOv8n**, theo vết từng xe bằng **ByteTrack**, đếm lượt xe **IN/OUT** qua vạch đếm, **ước lượng tốc độ (km/h)** bằng biến đổi phối cảnh, **phát hiện xe vượt tốc độ**, hiển thị thống kê theo thời gian thực, và xuất **video đã chú thích** cùng **báo cáo JSON**.

Đây là sản phẩm cuối (Bài 9) của giáo trình [Supervision](https://github.com/roboflow/supervision), được triển khai theo kiến trúc phân tầng Core / Pipeline / Services / UI.

![Traffic Monitor đang xử lý video trên GPU](docs/images/app-running.jpg)

---

## Mục lục

1. [Mô tả dự án](#1-mô-tả-dự-án)
2. [Công nghệ sử dụng](#2-công-nghệ-sử-dụng)
3. [Tính năng](#3-tính-năng)
4. [Kiến trúc hệ thống](#4-kiến-trúc-hệ-thống)
5. [Cài đặt và chạy](#5-cài-đặt-và-chạy)
6. [Hướng dẫn sử dụng](#6-hướng-dẫn-sử-dụng)
7. [Nghiệp vụ hệ thống](#7-nghiệp-vụ-hệ-thống)
8. [Thuật toán và quy tắc tính toán](#8-thuật-toán-và-quy-tắc-tính-toán)
9. [Cấu hình](#9-cấu-hình)
10. [Đầu ra](#10-đầu-ra)
11. [Kiểm thử](#11-kiểm-thử)
12. [Hiệu năng và ghi chú kỹ thuật](#12-hiệu-năng-và-ghi-chú-kỹ-thuật)

---

## 1. Mô tả dự án

### Bài toán

Camera giao thông ghi lại lượng video lớn, nhưng đếm xe và kiểm tra tốc độ thủ công thì tốn công và dễ sai. Dự án tự động hóa việc này từ một file video:

| Câu hỏi | Hệ thống trả lời bằng |
|---|---|
| Có bao nhiêu xe đi qua, theo mỗi chiều? | Đếm IN/OUT qua vạch đếm (LineZone) |
| Mỗi chiều có bao nhiêu xe con, xe máy, xe buýt, xe tải? | Đếm IN/OUT theo từng loại xe |
| Mỗi xe chạy nhanh bao nhiêu? | Tốc độ km/h tính từ tọa độ thực (mét) sau biến đổi phối cảnh |
| Xe nào vượt tốc độ cho phép? | Danh sách vi phạm: ID xe, loại xe, tốc độ tối đa, thời điểm |
| Làm sao lưu lại bằng chứng và số liệu? | Video MP4 có chú thích và báo cáo JSON |

### Đối tượng sử dụng

Sinh viên và kỹ sư Computer Vision học pipeline phát hiện → theo vết → phân tích, cũng như những ai cần công cụ desktop nhẹ để phân tích nhanh video giao thông mà không phải viết code.

### Phạm vi

- **Có:** xử lý file video (MP4/AVI/MOV/MKV), 4 loại xe COCO, hiệu chỉnh phối cảnh riêng cho từng video, chạy trên GPU (CUDA) và tự chuyển sang CPU khi không có GPU.
- **Chưa có:** camera trực tiếp (RTSP), nhận dạng biển số, lưu vào cơ sở dữ liệu.

---

## 2. Công nghệ sử dụng

| Nhóm | Công nghệ | Phiên bản đã kiểm thử | Vai trò trong dự án |
|---|---|---|---|
| Ngôn ngữ | **Python** | 3.10 | Toàn bộ ứng dụng; dùng type hints, dataclasses, `X \| None` |
| Phát hiện đối tượng | **Ultralytics YOLOv8n** (`yolov8n.pt`) | ultralytics 8.3.246 | Phát hiện xe (car, motorcycle, bus, truck) trên từng frame |
| Deep learning / GPU | **PyTorch + CUDA** | torch 2.5.1 + CUDA 12.1 | Chạy YOLO trên GPU NVIDIA với FP16 (half precision) |
| Computer Vision | **Supervision** (Roboflow) | 0.30.2 | `sv.Detections`, `ByteTrack`, `DetectionsSmoother`, `LineZone`, các annotator |
| Xử lý ảnh / video | **OpenCV** | 4.12 | Đọc/ghi video, `getPerspectiveTransform`, vẽ chú thích |
| Tính toán | **NumPy** | 2.2 | Ma trận tọa độ, lọc detections |
| Giao diện desktop | **PySide6 (Qt 6)** | 6.11 | Cửa sổ, widget, `QThread`, signal/slot, stylesheet tối |
| Đa luồng | `QThread` + `threading` + `queue` | – | Tách UI, inference, đọc video và ghi video ra các luồng riêng |
| Dữ liệu | `json`, `dataclasses` | stdlib | Cấu hình, hiệu chỉnh, báo cáo, model dữ liệu |
| Logging | `logging` (RotatingFileHandler) | stdlib | `outputs/logs/app.log` |
| Kiểm thử | **pytest** | 9.1 | 102 test: unit, GPU, pipeline, luồng UI end-to-end |

**Thuật toán chính:** YOLOv8 (one-stage detector), ByteTrack (multi-object tracking bằng Kalman filter kết hợp ghép cặp IoU), homography 4 điểm (perspective transform), ước lượng vận tốc trên cửa sổ thời gian trượt.

---

## 3. Tính năng

- 🎯 **Phát hiện và phân loại xe** bằng YOLOv8n trên GPU, lọc theo ngưỡng confidence (mặc định 0.30) và theo loại xe được chọn.
- 🔗 **Theo vết đa đối tượng** bằng ByteTrack: mỗi xe giữ một `tracker_id` ổn định, box được làm mượt để giảm rung.
- ➕ **Đếm IN/OUT** tổng và theo 4 loại xe qua `sv.LineZone` (anchor `BOTTOM_CENTER`).
- 📏 **Ước lượng tốc độ km/h** bằng biến đổi phối cảnh từ pixel sang mét, lưu lịch sử vị trí trong `deque`.
- 🚨 **Phát hiện vượt tốc độ**: box đỏ, nhãn `OVERSPEED`, danh sách vi phạm ghi tốc độ tối đa của từng xe.
- 🖼️ **Chú thích trên frame**: bounding box, loại xe, confidence, ID, vết di chuyển, tốc độ, vạch đếm, số IN/OUT, bảng thống kê trong frame.
- 📊 **Dashboard realtime**: Total IN/OUT, IN/OUT theo loại, số vi phạm, tốc độ trung bình và tối đa, số xe trong frame.
- ⏯️ **Điều khiển**: Open Video, Start, Pause, Resume, Stop, Export Report, Settings (có phím tắt).
- ⚙️ **Settings trên UI**: model, GPU/CPU, FP16, confidence, IoU, loại xe, giới hạn tốc độ, 4 điểm phối cảnh, kích thước đường, vạch đếm, đường dẫn đầu ra. Có ảnh xem trước cập nhật ngay khi chỉnh.
- 💾 **Tự động xuất** video MP4 đã chú thích và báo cáo JSON khi Stop hoặc khi hết video.
- 🛡️ **Xử lý lỗi thân thiện**: video không tồn tại hoặc hỏng, thiếu model, lỗi codec, cấu hình sai… đều hiện thông báo tiếng Việt, không để lộ traceback cho người dùng.
- 🔒 **Máy trạng thái**: chặn thao tác sai (Start khi chưa chọn video, Pause khi chưa chạy, Export khi chưa có kết quả…) và giải thích lý do.

| Hiệu chỉnh phối cảnh và vạch đếm | Kết thúc phiên (STOPPED) |
|---|---|
| ![Settings - Hiệu chỉnh](docs/images/settings-calibration.jpg) | ![Kết quả sau khi Stop](docs/images/app-stopped.jpg) |

---

## 4. Kiến trúc hệ thống

```text
┌──────────────────────── UI (PySide6) ──────────────────────────┐
│ MainWindow · VideoWidget · DashboardWidget · ControlPanel      │
│ SettingsDialog · VideoWorker (QThread) · ModelWarmupWorker     │
└──────────────┬─────────────────────────────────────────────────┘
               │ signals/slots (frame_ready, stats_ready, ...)
┌──────────────▼──────────── Pipeline ───────────────────────────┐
│ VideoProcessor: đọc ─▶ xử lý ─▶ ghi (2 luồng I/O, queue có giới hạn) │
│ FrameProcessor: TrafficMonitor + annotators                    │
└──────────────┬─────────────────────────────────────────────────┘
┌──────────────▼──────────── Core ───────────────────────────────┐
│ TrafficMonitor (chỉ điều phối)                                 │
│  ├─ VehicleDetector        YOLOv8n → sv.Detections → lọc       │
│  ├─ VehicleTracker         ByteTrack + DetectionsSmoother      │
│  ├─ VehicleCounter         LineZone IN/OUT theo loại xe        │
│  ├─ SpeedEstimator         homography + deque → km/h           │
│  └─ SpeedViolationDetector speed > limit                       │
└────────────────────────────────────────────────────────────────┘
 Services: ReportService · VideoService · ConfigService
 Nền dùng chung: models (dataclass, AppState) · config · constants · utils
```

**Pipeline cho mỗi frame:** Frame → Detection → Filtering → Tracking → Counting → Speed Estimation → Violation Detection → Annotation → Statistics → Processed Frame

**Luồng thread:**

```text
Main UI Thread ──start/pause/resume/stop──▶ VideoWorker (QThread)
      ▲                                        ├─ reader thread : cv2.VideoCapture → queue
      │  frame_ready(QImage)                   ├─ YOLO (GPU) → ByteTrack → LineZone → Speed → Annotate
      └──stats_ready / run_finished / error ───┴─ writer thread : queue → cv2.VideoWriter
```

### Cấu trúc thư mục

```text
traffic_monitor/
├── main.py                     # điểm khởi chạy: python main.py
├── requirements.txt
├── app/
│   ├── config/                 # settings.py (schema), default_config.json
│   ├── core/                   # detector, tracker, counter, speed_estimator,
│   │                           # violation_detector, traffic_monitor, exceptions
│   ├── models/                 # detection_result, traffic_statistics, violation, app_state
│   ├── pipeline/               # frame_processor, video_processor
│   ├── services/               # report_service, video_service, config_service
│   ├── ui/                     # main_window, video_widget, dashboard_widget, control_panel,
│   │                           # settings_dialog, video_worker, styles
│   ├── utils/                  # display, logger, file_utils, time_utils, geometry, device, qt_compat
│   └── constants/              # classes (COCO ids, màu), messages (thông báo tiếng Việt)
├── models/                     # yolov8n.pt (không commit, xem phần cài đặt)
├── data/
│   ├── input/                  # video đầu vào (không commit)
│   └── calibration/            # hiệu chỉnh theo video, ví dụ vehicles.json
├── outputs/                    # videos/, reports/, logs/
├── docs/                       # traffic_monitor_spec.html (tài liệu thiết kế), images/
└── tests/                      # 102 test pytest
```

Hướng phụ thuộc chỉ đi một chiều: `ui → pipeline/services → core → models/constants/utils`. Tầng `core` và `pipeline` không import PySide6, nên kiểm thử được mà không cần giao diện.

📄 Tài liệu thiết kế chi tiết (SDD, có demo UI tương tác): [`docs/traffic_monitor_spec.html`](docs/traffic_monitor_spec.html)

---

## 5. Cài đặt và chạy

### Yêu cầu

- Python **3.10+**
- GPU NVIDIA có CUDA (khuyến nghị). Đã kiểm thử trên GeForce MX570 2 GB. Máy không có GPU vẫn chạy được bằng CPU.
- Windows, Linux hoặc macOS

### Các bước

```bash
git clone https://github.com/Tuyetmai0705/Traffic-Monitor-YoloV8.git
cd Traffic-Monitor-YoloV8

# 1. PyTorch bản CUDA (chọn đúng bản CUDA của máy tại https://pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Các thư viện còn lại
pip install -r requirements.txt

# 3. Trọng số YOLOv8n → models/yolov8n.pt
python -c "from ultralytics import YOLO; YOLO('models/yolov8n.pt')"

# 4. Video mẫu → data/input/vehicles.mp4 (đã có sẵn hiệu chỉnh trong data/calibration/)
cd data/input && python -c "from supervision.assets import download_assets, VideoAssets; download_assets(VideoAssets.VEHICLES)" && cd ../..

# 5. Chạy ứng dụng
python main.py
```

Kiểm tra GPU: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

---

## 6. Hướng dẫn sử dụng

1. **Open Video** (Ctrl+O): chọn file trong `data/input/`. Khung video hiện frame đầu kèm vạch đếm (vàng) và vùng phối cảnh ABCD (hồng).
2. **Settings** (Ctrl+,): chỉnh model, thiết bị, confidence, giới hạn tốc độ, loại xe. Với video mới, vào tab **Hiệu chỉnh** để đặt 4 điểm A–D bao một đoạn đường có kích thước thực đã biết, rồi đặt vạch đếm.
3. **Start** (F5): bắt đầu xử lý trên GPU. Video chú thích và dashboard cập nhật liên tục.
4. **Pause** (F6) / **Resume** (F7): tạm dừng hoặc tiếp tục.
5. **Stop** (F8): dừng. Video và báo cáo được xuất tự động vào `outputs/`.
6. **Export Report** (Ctrl+E): lưu báo cáo JSON ra vị trí khác.

| Vùng giao diện | Nội dung |
|---|---|
| Video | Frame đã chú thích: box, loại xe, confidence, `#ID`, vết di chuyển, km/h, `OVERSPEED`, vạch đếm, IN/OUT, bảng thống kê trong frame |
| Dashboard | TOTAL IN/OUT, bảng IN/OUT theo loại, Violations, Average/Maximum speed, Speed limit, số xe trong frame, danh sách vi phạm |
| Thanh trạng thái | Trạng thái (màu), thông báo, file, GPU, FPS, thời gian xử lý mỗi frame, frame hiện tại/tổng, thanh tiến độ |

---

## 7. Nghiệp vụ hệ thống

### 7.1 Máy trạng thái

```text
IDLE ──open──▶ VIDEO_SELECTED ──start──▶ RUNNING ◀──resume── PAUSED
  │                  ▲                   │  │  └────pause─────▶  │
  └─mở lỗi─▶ ERROR   │                   │  └────stop──▶ STOPPED ◀─stop─┘
             │       │                   └──hết video──▶ COMPLETED
             └───────┴──── open (từ STOPPED / COMPLETED / ERROR)
```

| Trạng thái | Open | Start | Pause | Resume | Stop | Export | Settings |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| IDLE | ✔ | ✘ | ✘ | ✘ | ✘ | ✘ | ✔ |
| VIDEO_SELECTED | ✔ | ✔ | ✘ | ✘ | ✘ | ✘ | ✔ |
| RUNNING | ✘ | ✘ | ✔ | ✘ | ✔ | ✘ | ✘ |
| PAUSED | ✘ | ✘ | ✘ | ✔ | ✔ | ✘ | ✘ |
| STOPPED | ✔ | ✔ (chạy lại) | ✘ | ✘ | ✘ | ✔ | ✔ |
| COMPLETED | ✔ | ✔ (chạy lại) | ✘ | ✘ | ✘ | ✔ | ✔ |
| ERROR | ✔ | ✘ | ✘ | ✘ | ✘ | ✘ | ✔ |

Nút bị chặn hiển thị mờ nhưng vẫn bấm được; khi bấm, thanh trạng thái giải thích lý do. Luật nằm ở `app/models/app_state.py`.

### 7.2 Các luồng nghiệp vụ

| # | Luồng | Điều kiện | Các bước | Kết quả | Ngoại lệ |
|---|---|---|---|---|---|
| NV1 | Khởi động | – | Ghi log → đọc config → tạo `outputs/` → hiện cửa sổ → tải YOLOv8n lên GPU ở luồng nền | IDLE, "Model sẵn sàng trên cuda:0" | Không có GPU: chạy CPU và cảnh báo |
| NV2 | Mở video | Không đang chạy | Chọn file → kiểm tra file, codec, frame đầu → nạp hiệu chỉnh riêng → xem trước | VIDEO_SELECTED | File không tồn tại hoặc hỏng: ERROR. Chưa có hiệu chỉnh: đặt vạch mặc định và nhắc người dùng |
| NV3 | Cấu hình | Không đang chạy | Settings → chỉnh → Save | Lưu `user_config.json` và hiệu chỉnh theo video | Giá trị sai: tô đỏ ô sai, không lưu |
| NV4 | Xử lý | Đã chọn video | Start → worker QThread chạy toàn bộ pipeline trên GPU | RUNNING, UI vẫn phản hồi | Thiếu model hoặc lỗi inference: ERROR, vẫn lưu phần kết quả đã xử lý |
| NV5 | Tạm dừng / tiếp tục | RUNNING / PAUSED | Pause / Resume | PAUSED ↔ RUNNING | – |
| NV6 | Dừng | RUNNING / PAUSED | Stop → đóng video → xuất báo cáo | STOPPED | – |
| NV7 | Hoàn tất | Hết video | Tự động xuất video và báo cáo | COMPLETED | – |
| NV8 | Xuất báo cáo | STOPPED / COMPLETED | Export Report → chọn nơi lưu | File JSON mới | Không ghi được: báo lỗi |
| NV9 | Chạy lại | STOPPED / COMPLETED | Start → reset thống kê → chạy lại từ đầu | RUNNING | – |
| NV10 | Đóng ứng dụng | Bất kỳ | Dừng worker, chờ thread kết thúc | Thoát sạch, file video hợp lệ | – |

---

## 8. Thuật toán và quy tắc tính toán

| Bước | Quy tắc |
|---|---|
| Lọc | Giữ `confidence ≥ threshold` (mặc định 0.30) và `class_id ∈ {2 car, 3 motorcycle, 5 bus, 7 truck}` |
| Theo vết | `ByteTrack(frame_rate = fps thực của video)` → `DetectionsSmoother(length=5)` |
| Đếm | `LineZone.trigger()` được gọi **đúng 1 lần/frame**, anchor `BOTTOM_CENTER`. Đếm theo loại lấy từ mask `crossed_in` / `crossed_out`. Với vạch đi từ trái sang phải: xe đi lên là IN, xe đi xuống là OUT |
| Phối cảnh | `cv2.getPerspectiveTransform(ABCD_pixel → [0,0],[W,0],[W,L],[0,L] mét)` |
| Tốc độ | `v = ‖P_cuối − P_đầu‖ ÷ ((f_cuối − f_đầu) ÷ fps) × 3.6` trên `deque(maxlen = fps × window_seconds)`, cần tối thiểu `min_samples` điểm; **không** tính từ 2 frame liên tiếp |
| Vi phạm | `speed > limit` (bằng giới hạn thì không tính). Mỗi xe có 1 bản ghi, cập nhật `max_speed_kmh` |
| Tốc độ trung bình | Trung bình của tốc độ gần nhất của từng xe trong phiên |

---

## 9. Cấu hình

Cấu hình mặc định nằm ở `app/config/default_config.json`. Khi bấm Save trong Settings, cấu hình được ghi vào `app/config/user_config.json`. Hiệu chỉnh riêng từng video lưu tại `data/calibration/<tên_video>.json`.

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `model.path` | `models/yolov8n.pt` | Trọng số YOLO |
| `model.device` / `model.half` | `cuda:0` / `true` | Thiết bị suy luận và chế độ FP16 |
| `model.confidence` | `0.30` | Ngưỡng confidence |
| `classes` | `[2, 3, 5, 7]` | Loại xe cần theo dõi |
| `speed.limit_kmh` | `80` | Giới hạn tốc độ |
| `speed.window_seconds` / `min_samples` | `1.0` / `12` | Cửa sổ tính tốc độ |
| `perspective.source_points` | `[[1252,787],[2298,803],[5039,2159],[-550,2159]]` | 4 điểm A, B, C, D (pixel gốc) |
| `perspective.road_width_m` / `road_length_m` | `25` / `250` | Kích thước thực của vùng ABCD |
| `counting_line.start` / `end` | `[0,1500]` / `[3840,1500]` | Vạch đếm (pixel gốc) |
| `processing.max_width` | `1920` | Frame rộng hơn mức này được thu nhỏ trước khi xử lý; tọa độ được scale tự động |
| `output.*` | `outputs/...` | Đường dẫn video, báo cáo, log |

---

## 10. Đầu ra

```text
outputs/videos/traffic_result.mp4     video đã chú thích (codec mp4v; nếu không được thì avc1, XVID, MJPG)
outputs/reports/traffic_report.json   báo cáo
outputs/logs/app.log                  log (xoay vòng 5 MB × 3)
```

Ví dụ `traffic_report.json`:

```json
{
    "video": "data/input/vehicles.mp4",
    "total_in": 2,
    "total_out": 3,
    "by_class": {
        "car": { "in": 2, "out": 2 },
        "motorcycle": { "in": 0, "out": 0 },
        "bus": { "in": 0, "out": 0 },
        "truck": { "in": 0, "out": 1 }
    },
    "speed_violations": {
        "8": { "class": "car", "max_speed_kmh": 140.7, "speed_limit_kmh": 80, "first_seen": "00:00:01.3", "first_frame": 33, "last_frame": 60 }
    },
    "generated_at": "2026-09-28T14:04:24",
    "status": "stopped",
    "frames_processed": 105,
    "duration_seconds": 4.2,
    "model": "yolov8n.pt",
    "device": "cuda:0 (NVIDIA GeForce MX570 A)",
    "confidence": 0.3,
    "speed_limit_kmh": 80,
    "speed_summary": { "average_kmh": 107.5, "max_kmh": 140.7 },
    "output_video": "outputs/videos/traffic_result.mp4"
}
```

Năm khóa bắt buộc (`video`, `total_in`, `total_out`, `by_class`, `speed_violations`) luôn đứng đầu báo cáo.

---

## 11. Kiểm thử

```bash
python -m pytest                              # toàn bộ 102 test (~1 phút trên GPU)
python -m pytest -m "not gpu and not ui"      # chỉ chạy test nhanh, không cần GPU
```

| File | Nội dung kiểm thử |
|---|---|
| `test_config.py` | Đọc, kiểm tra hợp lệ, lưu cấu hình và hiệu chỉnh theo video |
| `test_detector.py` | Khởi tạo, lọc confidence và loại xe, lỗi thiếu model, **YOLOv8n thật trên GPU** |
| `test_tracker.py` | Gán `tracker_id`, ID ổn định qua các frame, reset |
| `test_counter.py` | IN/OUT theo chiều, theo loại xe, `trigger()` đúng 1 lần/frame |
| `test_speed.py` | Homography, 20 m/s → 72 km/h, thiếu mẫu, frame bị mất, điểm phối cảnh không hợp lệ |
| `test_violation.py` | Luật `>` giới hạn, 1 bản ghi/xe, cập nhật tốc độ tối đa |
| `test_traffic_monitor.py` | Điều phối pipeline, gắn số frame vào lỗi inference |
| `test_pipeline.py` | Đọc/ghi video, pause/resume/stop đa luồng, **pipeline GPU trên video thật** |
| `test_report.py` | Cấu trúc báo cáo JSON, tự tạo thư mục khi xuất |
| `test_state.py` | Luật chuyển trạng thái và thông báo chặn |
| `test_ui_flows.py` | **10 luồng nghiệp vụ end-to-end** trên MainWindow thật |

---

## 12. Hiệu năng và ghi chú kỹ thuật

- **Hiệu năng** (GeForce MX570 2 GB, video 4K xử lý ở 1920 px, YOLOv8n FP16): khoảng 12 fps khi chạy pipeline không UI, khoảng 8–11 fps khi chạy trong ứng dụng.
- **Đa luồng I/O:** đọc và ghi video chạy ở 2 luồng riêng, nối với nhau bằng queue có giới hạn, nên video không bao giờ bị nạp toàn bộ vào RAM.
- **Cache model:** YOLO chỉ được tải một lần cho mỗi tiến trình. Việc khởi động GPU chạy ở nền ngay khi mở app, nên frame đầu tiên xuất hiện khoảng 0,7 giây sau khi bấm Start.
- **PySide6 và torch:** `app/ui/__init__.py` nạp torch/ultralytics trước PySide6 và thay hàm kiểm tra import của PySide6 (`app/utils/qt_compat.py`). Nếu không, lần tải YOLO đầu tiên trong QThread mất hơn 30 giây.
- **Chú thích ASCII:** nhãn trên frame chỉ dùng ký tự ASCII vì `cv2.putText` không vẽ được Unicode (tiếng Việt).
- **Supervision:** `ByteTrack` bị đánh dấu deprecated từ bản 0.28 và sẽ bị gỡ ở 0.31, nên requirements ghim `supervision<0.31`.

---

## Tác giả

**Tuyetmai** · [github.com/Tuyetmai0705](https://github.com/Tuyetmai0705)

Dự án học tập dựa trên giáo trình Supervision (Roboflow), dùng mô hình YOLOv8 của Ultralytics.
