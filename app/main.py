"""FastAPI edge control plane and telemetry ingress for SpatialFlux."""

from __future__ import annotations

import hmac
import ipaddress
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status

from app.buffer import AsyncFrameBuffer, FrameDecodeError
from app.config import settings
from app.detector import LocalSpatialEvaluator
from app.dispatch import CloudVLARouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

frame_buffer = AsyncFrameBuffer()
spatial_evaluator = LocalSpatialEvaluator()
cloud_router = CloudVLARouter()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Close outbound cloud resources during server shutdown."""
    yield
    await cloud_router.aclose()


app = FastAPI(
    title="SpatialFlux",
    version="1.0.0",
    description="Edge-to-cloud video streaming ingestion and perception fabric.",
    lifespan=lifespan,
)


@dataclass(slots=True)
class PipelineTelemetry:
    """Mutable edge pipeline telemetry counters."""

    total_processed_frames: int = 0
    total_skipped_cloud_uploads: int = 0
    total_triggered_structural_drift_incidents: int = 0
    total_failed_frames: int = 0
    last_frame_latency_seconds: float = 0.0
    last_dispatch_latency_seconds: float = 0.0


telemetry = PipelineTelemetry()


def _is_loopback(host: str | None) -> bool:
    try:
        return host is not None and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_api_key(request: Request) -> None:
    """Gate ingest/telemetry.

    With API_KEY set, every request must present it (``X-API-Key`` or
    ``Authorization: Bearer``). Without a key, only loopback clients are served
    unless ALLOW_UNAUTHENTICATED is explicitly enabled.
    """
    key = settings.API_KEY
    if key:
        presented = request.headers.get("x-api-key", "")
        auth = request.headers.get("authorization", "")
        if not presented and auth.lower().startswith("bearer "):
            presented = auth[7:].strip()
        if hmac.compare_digest(presented.encode(), key.encode()):
            return
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")
    if settings.ALLOW_UNAUTHENTICATED:
        return
    client = request.client.host if request.client else None
    if _is_loopback(client):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="API_KEY is not configured; only loopback clients are allowed",
    )


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    """Read at most ``limit`` bytes; 413 if the upload is larger."""
    if file.size is not None and file.size > limit:
        raise HTTPException(
            status_code=413,
            detail=f"frame exceeds {limit} bytes",
        )
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"frame exceeds {limit} bytes",
        )
    return data


@app.post("/v1/spatial/ingest", dependencies=[Depends(require_api_key)])
async def ingest_spatial_frame(file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
    """Ingest a camera frame, evaluate drift, and route the payload to the cloud fabric."""
    frame_id = f"frame-{uuid4()}"
    start = time.perf_counter()
    try:
        frame_bytes = await _read_limited(file, settings.MAX_UPLOAD_BYTES)
        baseline_record, stored = await frame_buffer.push_frame(frame_id, frame_bytes)
        current_frame, ingest_timestamp = stored.frame, stored.ingest_timestamp
        if baseline_record is None:
            drift_score = 0.0
            drift_regions: list[dict[str, Any]] = []
        else:
            baseline_frame = baseline_record.frame
            drift_evaluation = await spatial_evaluator.compute_drift(
                current_frame=current_frame,
                baseline_frame=baseline_frame,
            )
            drift_score = drift_evaluation.drift_score
            drift_regions = [
                {
                    "x": region.x,
                    "y": region.y,
                    "width": region.width,
                    "height": region.height,
                    "area_ratio": region.area_ratio,
                }
                for region in drift_evaluation.regions
            ]

        dispatch_response = await cloud_router.route_spatial_payload(
            frame_id=frame_id,
            frame_data=current_frame,
            drift_score=drift_score,
        )
        telemetry.total_processed_frames += 1
        telemetry.last_frame_latency_seconds = time.perf_counter() - start
        telemetry.last_dispatch_latency_seconds = float(
            dispatch_response.get("elapsed_seconds", 0.0)
        )

        if dispatch_response.get("cloud_upload_skipped") is True:
            telemetry.total_skipped_cloud_uploads += 1
        if dispatch_response.get("frame_state") == "ANOMALY_TRIGGERED":
            telemetry.total_triggered_structural_drift_incidents += 1

        logger.info(
            "spatial_ingest_completed",
            extra={
                "frame_id": frame_id,
                "drift_score": drift_score,
                "frame_state": dispatch_response.get("frame_state"),
                "latency_seconds": telemetry.last_frame_latency_seconds,
            },
        )
        return {
            "frame_id": frame_id,
            "ingest_timestamp": ingest_timestamp,
            "drift_score": drift_score,
            "drift_regions": drift_regions,
            "dispatch": dispatch_response,
        }
    except HTTPException:
        telemetry.total_failed_frames += 1
        raise
    except FrameDecodeError as exc:
        telemetry.total_failed_frames += 1
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        telemetry.total_failed_frames += 1
        logger.exception("spatial_ingest_runtime_failure", extra={"frame_id": frame_id})
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        telemetry.total_failed_frames += 1
        logger.exception("spatial_ingest_unhandled_failure", extra={"frame_id": frame_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="spatial ingest failed",
        ) from exc


@app.get("/v1/spatial/telemetry", dependencies=[Depends(require_api_key)])
async def get_spatial_telemetry() -> dict[str, Any]:
    """Return current ingestion, dispatch, and latency telemetry."""
    try:
        return {
            "total_processed_frames_count": telemetry.total_processed_frames,
            "total_skipped_cloud_uploads": telemetry.total_skipped_cloud_uploads,
            "total_triggered_structural_drift_incidents": (
                telemetry.total_triggered_structural_drift_incidents
            ),
            "total_failed_frames": telemetry.total_failed_frames,
            "current_frame_drop_latency_profiles": {
                "buffer_dropped_frames": frame_buffer.dropped_frames,
                "buffer_retained_frames": frame_buffer.retained_frames,
                "last_buffer_ingest_latency_seconds": frame_buffer.last_ingest_latency_seconds,
                "last_pipeline_latency_seconds": telemetry.last_frame_latency_seconds,
                "last_dispatch_latency_seconds": telemetry.last_dispatch_latency_seconds,
            },
        }
    except Exception as exc:
        logger.exception("telemetry_read_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="telemetry read failed",
        ) from exc
