"""SpatialFlux edge-to-cloud video ingestion and perception fabric."""

import os

# OpenCV reads its decode pixel cap once, before the first decode. A tiny PNG
# can declare a 16k x 16k canvas (~770 MB decoded), so cap it here, before any
# module imports cv2, at the same limit the buffer enforces (MAX_FRAME_PIXELS).
os.environ.setdefault(
    "OPENCV_IO_MAX_IMAGE_PIXELS", os.environ.get("MAX_FRAME_PIXELS", str(7680 * 4320))
)

__all__ = [
    "__version__",
]

__version__ = "1.0.1"
