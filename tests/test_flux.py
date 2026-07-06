"""Distributed SpatialFlux integration tests."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import cv2
import httpx
import numpy as np
import pytest

from app.detector import LocalSpatialEvaluator
from app.dispatch import CloudVLARouter


def _mock_response(payload: dict[str, object]) -> httpx.Response:
    return httpx.Response(status_code=200, json=payload, request=httpx.Request("POST", "http://test"))


@pytest.mark.asyncio
async def test_nominal_stability_flow_skips_high_fidelity_upload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    evaluator = LocalSpatialEvaluator()
    baseline = np.full((96, 96, 3), 32, dtype=np.uint8)
    current = baseline.copy()
    current[0:4, 0:4] = 35

    drift_score = await evaluator.compute_structural_drift(current, baseline)

    post_mock = AsyncMock(return_value=_mock_response({"heartbeat": "accepted"}))
    async with CloudVLARouter(endpoint_url="http://test", client=httpx.AsyncClient()) as router:
        with patch.object(router._client, "post", post_mock):
            result = await router.route_spatial_payload(
                frame_id="nominal-001",
                frame_data=current,
                drift_score=drift_score,
            )

    assert drift_score < 0.15
    assert result["frame_state"] == "NOMINAL_STABLE"
    assert result["cloud_upload_skipped"] is True
    assert result["elapsed_seconds"] < 1.0
    assert post_mock.await_count == 1
    assert post_mock.await_args is not None
    _, kwargs = post_mock.await_args
    assert "json" in kwargs
    assert "files" not in kwargs
    assert kwargs["json"]["high_fidelity_upload"] is False
    assert "cost_optimized_skip" in caplog.text


@pytest.mark.asyncio
async def test_structural_anomaly_flow_uploads_jpeg_multipart() -> None:
    evaluator = LocalSpatialEvaluator()
    baseline = np.zeros((128, 128, 3), dtype=np.uint8)
    current = baseline.copy()
    cv2.rectangle(current, (12, 12), (116, 116), (255, 255, 255), thickness=-1)
    cv2.circle(current, (64, 64), 24, (0, 0, 0), thickness=-1)

    drift_score = await evaluator.compute_structural_drift(current, baseline)

    post_mock = AsyncMock(return_value=_mock_response({"motor_commands": [0.1, -0.2, 0.0]}))
    async with CloudVLARouter(endpoint_url="http://test", client=httpx.AsyncClient()) as router:
        with patch.object(router._client, "post", post_mock):
            result = await router.route_spatial_payload(
                frame_id="anomaly-001",
                frame_data=current,
                drift_score=drift_score,
            )

    assert drift_score > 0.15
    assert result["frame_state"] == "ANOMALY_TRIGGERED"
    assert result["cloud_upload_skipped"] is False
    assert result["payload_bytes"] > 0
    assert result["motor_commands"] == {"motor_commands": [0.1, -0.2, 0.0]}
    post_mock.assert_awaited_once()
    assert post_mock.await_args is not None
    _, kwargs = post_mock.await_args
    assert "files" in kwargs
    assert "data" in kwargs
    filename, payload_bytes, media_type = kwargs["files"]["file"]
    assert filename == "anomaly-001.jpg"
    assert media_type == "image/jpeg"
    assert isinstance(payload_bytes, bytes)
    decoded = cv2.imdecode(np.frombuffer(payload_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
