"""Audit fixes (Oct 2026): atomic baseline/stored frame pairing, upload size
limit, and API-key gate on ingest/telemetry."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.buffer import AsyncFrameBuffer
from app.config import settings

ROUTED: list[dict[str, Any]] = []


def _png(value: int, size: int = 32) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((size, size, 3), value, dtype=np.uint8))
    assert ok
    return bytes(buf.tobytes())


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "frame_buffer", AsyncFrameBuffer(max_frames=8))
    monkeypatch.setattr(settings, "API_KEY", "")
    monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED", False)
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
    ROUTED.clear()

    async def fake_route(**kw: Any) -> dict[str, Any]:
        ROUTED.append(kw)
        return {
            "frame_state": "NOMINAL_STABLE",
            "cloud_upload_skipped": True,
            "elapsed_seconds": 0.0,
        }

    monkeypatch.setattr(main.cloud_router, "route_spatial_payload", fake_route)


def _client(host: str = "127.0.0.1") -> TestClient:
    return TestClient(main.app, client=(host, 50000))


def _ingest(c: TestClient, data: bytes, **headers: str) -> Any:
    return c.post(
        "/v1/spatial/ingest", files={"file": ("f.png", data, "image/png")}, headers=headers
    )


# -- buffer atomicity ----------------------------------------------------------
@pytest.mark.asyncio
async def test_push_frame_returns_previous_and_stored() -> None:
    buf = AsyncFrameBuffer(max_frames=4)
    prev, stored = await buf.push_frame("a", _png(10))
    assert prev is None and stored.frame_id == "a"
    prev2, stored2 = await buf.push_frame("b", _png(20))
    assert prev2 is stored and stored2.frame_id == "b"
    latest = await buf.get_latest_frame()
    assert latest is not None and latest[0] == "b"


@pytest.mark.asyncio
async def test_concurrent_pushes_form_a_consistent_chain() -> None:
    buf = AsyncFrameBuffer(max_frames=64)
    results = await asyncio.gather(*(buf.push_frame(f"f{i}", _png(i)) for i in range(20)))
    assert len({s.frame_id for _, s in results}) == 20
    prevs = [p.frame_id if p else None for p, _ in results]
    assert prevs.count(None) == 1
    assert all(p != s.frame_id for p, (_, s) in zip(prevs, results, strict=True))
    assert len({p for p in prevs if p}) == 19


@pytest.mark.asyncio
async def test_drop_counter_still_counts_when_full() -> None:
    buf = AsyncFrameBuffer(max_frames=2)
    for i in range(3):
        await buf.push_frame(f"f{i}", _png(i))
    assert buf.dropped_frames == 1 and buf.retained_frames == 2


def test_ingest_routes_its_own_frame_with_baseline() -> None:
    c = _client()
    r1 = _ingest(c, _png(0))
    r2 = _ingest(c, _png(255))
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["drift_score"] == 0.0
    assert r2.json()["drift_score"] > 0.5
    assert [x["frame_id"] for x in ROUTED] == [r1.json()["frame_id"], r2.json()["frame_id"]]
    assert int(ROUTED[1]["frame_data"].mean()) == 255


# -- size limit -----------------------------------------------------------------
def test_oversized_upload_is_413(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 2048)
    r = _ingest(_client(), b"\x00" * 4096)
    assert r.status_code == 413
    assert main.frame_buffer.retained_frames == 0


def test_upload_at_limit_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _png(7)
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", len(data))
    assert _ingest(_client(), data).status_code == 200


@pytest.mark.asyncio
async def test_read_limited_stops_reading_past_limit() -> None:
    upload = AsyncMock()
    upload.size = None
    upload.read = AsyncMock(return_value=b"x" * 11)
    with pytest.raises(main.HTTPException) as exc:
        await main._read_limited(upload, 10)
    assert exc.value.status_code == 413
    upload.read.assert_awaited_once_with(11)
    upload.read = AsyncMock(return_value=b"x" * 10)
    assert await main._read_limited(upload, 10) == b"x" * 10


@pytest.mark.asyncio
async def test_read_limited_rejects_declared_size_without_reading() -> None:
    upload = AsyncMock()
    upload.size = 11
    with pytest.raises(main.HTTPException) as exc:
        await main._read_limited(upload, 10)
    assert exc.value.status_code == 413
    upload.read.assert_not_awaited()
    upload.size = 10
    upload.read = AsyncMock(return_value=b"y" * 10)
    assert await main._read_limited(upload, 10) == b"y" * 10


# -- auth -------------------------------------------------------------------------
def test_no_key_allows_loopback_only() -> None:
    assert _ingest(_client("127.0.0.1"), _png(1)).status_code == 200
    assert _client("::1").get("/v1/spatial/telemetry").status_code == 200
    assert _ingest(_client("203.0.113.5"), _png(1)).status_code == 401
    assert _client("203.0.113.5").get("/v1/spatial/telemetry").status_code == 401
    assert _client("not-an-ip").get("/v1/spatial/telemetry").status_code == 401


def test_is_loopback_helper() -> None:
    assert main._is_loopback("127.0.0.1") is True
    assert main._is_loopback(None) is False
    assert main._is_loopback("garbage") is False


def test_allow_unauthenticated_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED", True)
    assert _client("203.0.113.5").get("/v1/spatial/telemetry").status_code == 200


def test_key_required_everywhere_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "API_KEY", "k3y")
    monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED", True)  # key still wins
    c = _client("127.0.0.1")  # loopback is not exempt once a key is set
    assert _ingest(c, _png(1)).status_code == 401
    assert _ingest(c, _png(1), **{"X-API-Key": "nope"}).status_code == 401
    assert _ingest(c, _png(1), **{"X-API-Key": "k3y"}).status_code == 200
    ok = c.get("/v1/spatial/telemetry", headers={"Authorization": "Bearer k3y"})
    assert ok.status_code == 200
    bad = c.get("/v1/spatial/telemetry", headers={"Authorization": "Bearer bad"})
    assert bad.status_code == 401
    assert main.frame_buffer.retained_frames == 1


def test_rejected_frames_count_as_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 1024)
    before = main.telemetry.total_failed_frames
    _ingest(_client(), b"\x00" * 2048)
    assert main.telemetry.total_failed_frames == before + 1


# -- mutation-testing follow-ups ---------------------------------------------------
def test_x_api_key_header_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "API_KEY", "k3y")
    r = _client().get(
        "/v1/spatial/telemetry", headers={"X-API-Key": "wrong", "Authorization": "Bearer k3y"}
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_stored_frame_metadata_is_sane() -> None:
    import time

    buf = AsyncFrameBuffer(max_frames=2)
    before = time.time()
    _, stored = await buf.push_frame("a", _png(3))
    assert before - 1 <= stored.ingest_timestamp <= time.time() + 1
    assert 0.0 <= buf.last_ingest_latency_seconds < 5.0


@pytest.mark.asyncio
async def test_push_frame_decode_and_internal_errors() -> None:
    from app.buffer import FrameDecodeError

    buf = AsyncFrameBuffer(max_frames=2)
    with pytest.raises(FrameDecodeError):
        await buf.push_frame("bad", b"not an image")
    with pytest.raises(RuntimeError):
        await buf.push_frame("bad", object())  # type: ignore[arg-type]
    assert buf.retained_frames == 0


def test_undecodable_frame_is_400_and_not_stored() -> None:
    r = _ingest(_client(), b"definitely not a png")
    assert r.status_code == 400
    assert main.frame_buffer.retained_frames == 0


def test_dispatch_failure_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(**kw: Any) -> dict[str, Any]:
        raise RuntimeError("cloud down")

    monkeypatch.setattr(main.cloud_router, "route_spatial_payload", boom)
    assert _ingest(_client(), _png(1)).status_code == 502


def test_unexpected_failure_is_500(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(**kw: Any) -> dict[str, Any]:
        raise KeyError("x")

    monkeypatch.setattr(main.cloud_router, "route_spatial_payload", boom)
    r = _ingest(_client(), _png(1))
    assert r.status_code == 500 and r.json()["detail"] == "spatial ingest failed"


def test_telemetry_counters_after_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    t = main.PipelineTelemetry()
    monkeypatch.setattr(main, "telemetry", t)

    async def anomaly(**kw: Any) -> dict[str, Any]:
        return {
            "frame_state": "ANOMALY_TRIGGERED",
            "cloud_upload_skipped": False,
            "elapsed_seconds": 0.25,
        }

    c = _client()
    _ingest(c, _png(1))  # nominal (fixture router): skipped upload
    monkeypatch.setattr(main.cloud_router, "route_spatial_payload", anomaly)
    _ingest(c, _png(200))
    assert t.total_processed_frames == 2
    assert t.total_skipped_cloud_uploads == 1
    assert t.total_triggered_structural_drift_incidents == 1
    assert t.last_dispatch_latency_seconds == 0.25
    assert 0.0 <= t.last_frame_latency_seconds < 30.0
    body = c.get("/v1/spatial/telemetry").json()
    assert body["total_processed_frames_count"] == 2


# -- decoded frame size (decompression bombs) ----------------------------------
def test_frame_above_pixel_limit_rejected_and_not_buffered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MAX_FRAME_PIXELS", 64 * 64)
    with _client() as c:
        r = _ingest(c, _png(10, size=65))
        assert r.status_code == 400
        assert "limit" in r.json()["detail"]
        assert main.frame_buffer.retained_frames == 0
        assert _ingest(c, _png(10, size=64)).status_code == 200
    assert len(ROUTED) == 1


def test_opencv_pixel_cap_applied_on_import() -> None:
    # A ~270 KB PNG declaring 16000x16000 must be refused by the decoder
    # itself (via OPENCV_IO_MAX_IMAGE_PIXELS), not decoded into ~770 MB.
    import os
    import subprocess
    import sys

    code = (
        "import app, cv2, numpy as np\n"
        "from app.buffer import AsyncFrameBuffer, FrameDecodeError\n"
        "ok, b = cv2.imencode('.png', np.zeros((16000, 16000), np.uint8))\n"
        "try:\n"
        "    AsyncFrameBuffer._decode_frame(bytes(b))\n"
        "except FrameDecodeError:\n"
        "    print('refused')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "OPENCV_IO_MAX_IMAGE_PIXELS"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env
    )
    assert out.stdout.strip() == "refused", out.stderr
