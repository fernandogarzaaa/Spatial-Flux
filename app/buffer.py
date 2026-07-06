"""High-throughput asynchronous frame ingestion and bounded retention."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import settings

logger = logging.getLogger(__name__)

FrameArray = NDArray[np.uint8]


class FrameDecodeError(ValueError):
    """Raised when inbound image bytes cannot be decoded into a frame."""


@dataclass(slots=True, frozen=True)
class BufferedFrame:
    """Decoded frame retained in the local edge ring buffer."""

    frame_id: str
    frame: FrameArray
    ingest_timestamp: float


class AsyncFrameBuffer:
    """Thread-safe async ring buffer for decoded camera frames."""

    def __init__(self, max_frames: int | None = None) -> None:
        self._frames: deque[BufferedFrame] = deque(maxlen=max_frames or settings.MAX_BUFFER_FRAMES)
        self._lock = asyncio.Lock()
        self._dropped_frames = 0
        self._last_ingest_latency_seconds = 0.0

    @property
    def dropped_frames(self) -> int:
        """Return how many retained frames have been overwritten by the ring buffer."""
        return self._dropped_frames

    @property
    def retained_frames(self) -> int:
        """Return the number of frames currently retained in memory."""
        return len(self._frames)

    @property
    def last_ingest_latency_seconds(self) -> float:
        """Return the latest decode-and-store latency in seconds."""
        return self._last_ingest_latency_seconds

    async def push_frame(self, frame_id: str, frame_bytes: bytes) -> None:
        """Decode encoded image bytes and append the frame to the bounded buffer."""
        start = time.perf_counter()
        try:
            frame = await asyncio.to_thread(self._decode_frame, frame_bytes)
            ingest_timestamp = time.time_ns() / 1_000_000_000
            async with self._lock:
                if len(self._frames) == self._frames.maxlen:
                    self._dropped_frames += 1
                self._frames.append(
                    BufferedFrame(
                        frame_id=frame_id,
                        frame=frame,
                        ingest_timestamp=ingest_timestamp,
                    )
                )
                self._last_ingest_latency_seconds = time.perf_counter() - start
            logger.info(
                "frame_ingested",
                extra={
                    "frame_id": frame_id,
                    "retained_frames": len(self._frames),
                    "latency_seconds": self._last_ingest_latency_seconds,
                },
            )
        except FrameDecodeError:
            logger.exception("frame_decode_failed", extra={"frame_id": frame_id})
            raise
        except Exception as exc:
            logger.exception("frame_ingest_failed", extra={"frame_id": frame_id})
            raise RuntimeError("failed to push frame into async buffer") from exc

    async def get_latest_frame(self) -> tuple[str, FrameArray, float] | None:
        """Return the newest retained frame without popping it from the buffer."""
        try:
            async with self._lock:
                if not self._frames:
                    return None
                latest = self._frames[-1]
                logger.debug("latest_frame_read", extra={"frame_id": latest.frame_id})
                return latest.frame_id, latest.frame, latest.ingest_timestamp
        except Exception as exc:
            logger.exception("latest_frame_read_failed")
            raise RuntimeError("failed to read latest frame from async buffer") from exc

    @staticmethod
    def _decode_frame(frame_bytes: bytes) -> FrameArray:
        if not frame_bytes:
            raise FrameDecodeError("empty frame payload cannot be decoded")
        encoded_buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(encoded_buffer, cv2.IMREAD_COLOR)
        if decoded is None:
            raise FrameDecodeError("OpenCV could not decode frame payload")
        return cast(FrameArray, decoded)
