"""Open video captures and create video writers with codec fallback."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from app.core.exceptions import OutputPathError, VideoCodecError, VideoNotFoundError, VideoOpenError
from app.utils.file_utils import ensure_parent_dir, resolve_path
from app.utils.logger import get_logger

logger = get_logger("video_service")

# (fourcc, extension) tried in order. mp4v ships with every OpenCV build.
WRITER_CODECS: tuple[tuple[str, str], ...] = (("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi"))
DEFAULT_FPS = 30.0


class VideoService:
    """File-level video I/O. Never loads a whole video into memory."""

    def open_capture(self, path: str | Path) -> cv2.VideoCapture:
        """Open a video and verify the first frame decodes; the capture is rewound to frame 0."""
        video_path = resolve_path(path)
        if not video_path.is_file():
            raise VideoNotFoundError(video_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise VideoOpenError(video_path)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise VideoCodecError(f"không đọc được frame đầu của {video_path.name}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return capture

    @staticmethod
    def get_video_info(capture: cv2.VideoCapture) -> sv.VideoInfo:
        """Read resolution, fps and frame count from an open capture."""
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 240:
            logger.warning("Invalid FPS %.2f reported, using %.0f", fps, DEFAULT_FPS)
            fps = DEFAULT_FPS
        return sv.VideoInfo(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(fps),
            total_frames=max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        )

    def probe(self, path: str | Path) -> tuple[sv.VideoInfo, np.ndarray]:
        """Validate a video and return its info plus the first frame (for preview)."""
        capture = self.open_capture(path)
        try:
            info = self.get_video_info(capture)
            _, frame = capture.read()
            return info, frame
        finally:
            capture.release()

    def create_writer(self, path: str | Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
        """Create a writer, trying codecs in order. Returns (writer, actual output path)."""
        try:
            target = ensure_parent_dir(path)
        except OSError as exc:
            raise OutputPathError(Path(path).parent, str(exc)) from exc
        for fourcc, extension in WRITER_CODECS:
            output = target if target.suffix.lower() == extension else target.with_suffix(extension)
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*fourcc), fps, size)
            if writer.isOpened():
                logger.info("Video writer opened: %s (%s, %dx%d @ %.1f fps)", output, fourcc, size[0], size[1], fps)
                return writer, output
            writer.release()
            logger.warning("Codec %s unavailable for %s", fourcc, output)
        raise VideoCodecError("không có codec nào ghi được video kết quả")
