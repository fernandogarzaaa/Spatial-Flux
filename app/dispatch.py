"""Selective high-fidelity VLA cloud routing."""

from __future__ import annotations

import asyncio
import logging
import time
from types import TracebackType
from typing import Any

import cv2
import httpx
import numpy as np
from numpy.typing import NDArray

from app.config import settings
from app.detector import FrameState, LocalSpatialEvaluator

logger = logging.getLogger(__name__)

FrameArray = NDArray[np.uint8]


class CloudVLARouter:
    """Async edge-to-cloud dispatch client for telemetry and anomaly frames."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        timeout_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._endpoint_url = endpoint_url or settings.CLOUD_VLA_ENDPOINT_URL
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> CloudVLARouter:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the owned HTTP client."""
        if self._owns_client:
            await self._client.aclose()

    async def route_spatial_payload(
        self,
        frame_id: str,
        frame_data: FrameArray,
        drift_score: float,
    ) -> dict[str, Any]:
        """Dispatch telemetry only for nominal frames and multipart JPEG for anomalies."""
        state = LocalSpatialEvaluator.classify_drift(drift_score)
        try:
            if state == "NOMINAL_STABLE":
                return await self._route_nominal_payload(frame_id, drift_score, state)
            return await self._route_anomaly_payload(frame_id, frame_data, drift_score, state)
        except httpx.HTTPError as exc:
            logger.exception(
                "cloud_vla_http_error",
                extra={"frame_id": frame_id, "drift_score": drift_score, "frame_state": state},
            )
            raise RuntimeError("cloud VLA dispatch failed") from exc
        except ValueError:
            logger.exception("cloud_vla_payload_validation_failed", extra={"frame_id": frame_id})
            raise
        except Exception as exc:
            logger.exception("cloud_vla_dispatch_failed", extra={"frame_id": frame_id})
            raise RuntimeError("cloud VLA dispatch failed") from exc

    async def _route_nominal_payload(
        self,
        frame_id: str,
        drift_score: float,
        state: FrameState,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        metadata = {
            "frame_id": frame_id,
            "drift_score": drift_score,
            "frame_state": state,
            "high_fidelity_upload": False,
            "route": "metadata_heartbeat_only",
        }
        response = await self._client.post(self._endpoint_url, json=metadata)
        response.raise_for_status()
        elapsed = time.perf_counter() - start
        logger.info(
            "cost_optimized_skip",
            extra={
                "frame_id": frame_id,
                "drift_score": drift_score,
                "elapsed_seconds": elapsed,
            },
        )
        return {
            "frame_id": frame_id,
            "frame_state": state,
            "drift_score": drift_score,
            "cloud_upload_skipped": True,
            "elapsed_seconds": elapsed,
            "cloud_response": self._safe_json(response),
        }

    async def _route_anomaly_payload(
        self,
        frame_id: str,
        frame_data: FrameArray,
        drift_score: float,
        state: FrameState,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        jpeg_bytes = await asyncio.to_thread(self._encode_jpeg, frame_data)
        data = {
            "frame_id": frame_id,
            "drift_score": str(drift_score),
            "frame_state": state,
        }
        files = {
            "file": (
                f"{frame_id}.jpg",
                jpeg_bytes,
                "image/jpeg",
            )
        }
        async with self._lock:
            response = await self._client.post(self._endpoint_url, data=data, files=files)
        response.raise_for_status()
        elapsed = time.perf_counter() - start
        logger.info(
            "high_fidelity_vla_upload",
            extra={
                "frame_id": frame_id,
                "drift_score": drift_score,
                "payload_bytes": len(jpeg_bytes),
                "elapsed_seconds": elapsed,
            },
        )
        return {
            "frame_id": frame_id,
            "frame_state": state,
            "drift_score": drift_score,
            "cloud_upload_skipped": False,
            "elapsed_seconds": elapsed,
            "payload_bytes": len(jpeg_bytes),
            "motor_commands": self._safe_json(response),
        }

    @staticmethod
    def _encode_jpeg(frame_data: FrameArray) -> bytes:
        if frame_data.size == 0:
            raise ValueError("cannot encode an empty frame")
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), settings.EDGE_COMPRESSION_QUALITY]
        success, encoded = cv2.imencode(".jpg", frame_data, encode_params)
        if not success:
            raise ValueError("OpenCV failed to encode anomaly frame as JPEG")
        return bytes(encoded.tobytes())

    @staticmethod
    def _safe_json(response: httpx.Response) -> dict[str, Any] | list[Any] | str:
        try:
            parsed = response.json()
        except ValueError:
            return response.text
        if isinstance(parsed, dict | list | str):
            return parsed
        return {"value": parsed}
