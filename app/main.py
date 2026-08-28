"""FastAPI edge control plane and telemetry ingress for SpatialFlux."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from app.buffer import AsyncFrameBuffer, FrameDecodeError
from app.detector import LocalSpatialEvaluator
from app.dispatch import CloudVLARouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SpatialFlux",
    version="1.0.0",
    description="Edge-to-cloud video streaming ingestion and perception fabric.",
)

frame_buffer = AsyncFrameBuffer()
spatial_evaluator = LocalSpatialEvaluator()
cloud_router = CloudVLARouter()


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


@app.on_event("shutdown")
async def shutdown_router() -> None:
    """Close outbound cloud resources during server shutdown."""
    await cloud_router.aclose()


@app.post("/v1/spatial/ingest")
async def ingest_spatial_frame(file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
    """Ingest a camera frame, evaluate drift, and route the payload to the cloud fabric."""
    frame_id = f"frame-{uuid4()}"
    start = time.perf_counter()
    try:
        baseline_record = await frame_buffer.get_latest_frame()
        frame_bytes = await file.read()
        await frame_buffer.push_frame(frame_id, frame_bytes)
        latest_record = await frame_buffer.get_latest_frame()
        if latest_record is None:
            raise RuntimeError("frame buffer did not retain the uploaded frame")

        _, current_frame, ingest_timestamp = latest_record
        if baseline_record is None:
            drift_score = 0.0
            drift_regions: list[dict[str, Any]] = []
        else:
            _, baseline_frame, _ = baseline_record
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


@app.get("/v1/spatial/telemetry")
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
