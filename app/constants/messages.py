"""User-facing messages (Vietnamese). Code never shows tracebacks to users."""

APP_TITLE = "TRAFFIC MONITOR"

# Chặn hành động sai trạng thái
BLOCK_START_NO_VIDEO = "Hãy chọn video trước khi bắt đầu."
BLOCK_START_RUNNING = "Video đang được xử lý."
BLOCK_PAUSE = "Chỉ tạm dừng được khi đang xử lý."
BLOCK_RESUME = "Chỉ tiếp tục được khi đang tạm dừng."
BLOCK_STOP = "Chưa có phiên xử lý nào để dừng."
BLOCK_EXPORT = "Chưa có kết quả. Hãy dừng xử lý hoặc chờ video chạy xong."
BLOCK_OPEN = "Hãy dừng xử lý trước khi mở video khác."
BLOCK_SETTINGS = "Dừng xử lý trước khi thay đổi cài đặt."

# Trạng thái xử lý
VIDEO_SELECTED = "Đã chọn video {name} ({width}x{height}, {fps:.0f} fps, {frames} frame)."
LOADING_MODEL = "Đang tải model {model} lên {device}..."
MODEL_READY = "Model sẵn sàng trên {device}."
WARMING_UP = "Đang khởi tạo GPU và tải model {model}..."
PROCESSING = "Đang xử lý..."
PAUSED = "Đã tạm dừng tại frame {frame}."
RESUMED = "Tiếp tục xử lý."
STOPPING = "Đang dừng và xuất kết quả..."
EXPORTED = "Đã xuất video {video} và báo cáo {report}."
REPORT_SAVED = "Đã lưu báo cáo: {path}"
SETTINGS_SAVED = "Đã lưu cài đặt."
CALIBRATION_SAVED = "Đã lưu hiệu chỉnh cho video: {path}"
CALIBRATION_LOADED = "Đã áp dụng hiệu chỉnh riêng cho video {name}."
CALIBRATION_MISSING = (
    "Video {name} chưa có hiệu chỉnh. Vạch đếm được đặt mặc định ở 70% chiều cao; "
    "hãy vào Settings để chỉnh perspective và vạch đếm cho đúng tốc độ."
)
GPU_FALLBACK = "Không tìm thấy GPU CUDA, đang chạy bằng CPU."

# Lỗi
VIDEO_NOT_FOUND = "Không tìm thấy video: {path}"
VIDEO_OPEN_FAILED = "Không mở được video {path}. File có thể bị hỏng."
VIDEO_CODEC_FAILED = "Định dạng video không được hỗ trợ ({detail}). Hãy chuyển sang MP4 (H.264)."
MODEL_NOT_FOUND = "Không tìm thấy model {path}. Chọn lại trong Settings."
MODEL_LOAD_FAILED = "Không tải được model YOLO {path}. Kiểm tra file .pt."
INFERENCE_FAILED = "Lỗi xử lý tại frame {frame}: {detail}. Kết quả đến frame này đã được lưu."
INVALID_CONFIG = "Cấu hình không hợp lệ ở '{field}': {reason}"
INVALID_PERSPECTIVE = "Perspective points không hợp lệ: {reason}"
OUTPUT_PATH_FAILED = "Không ghi được vào {path}: {detail}"
UNEXPECTED_ERROR = "Đã xảy ra lỗi không mong muốn. Chi tiết được ghi trong outputs/logs/app.log."
