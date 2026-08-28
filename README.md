# SpatialFlux

A lightweight edge frame-ingestion service that triages camera frames before they reach the cloud. SpatialFlux decodes incoming frames, compares each one against the previous frame using pixel-level drift detection, and routes the result to a configurable cloud endpoint — sending only lightweight metadata for stable ("nominal") frames, and the full JPEG for frames that cross an anomaly threshold. The intent is to let a downstream Vision-Language-Action (VLA) or vision service spend its inference budget on frames that actually changed, instead of every frame in the stream.

---

## What it does today

SpatialFlux is a FastAPI service (`app/main.py`) built from three components:

- **`app/buffer.py` — `AsyncFrameBuffer`**: an async, bounded ring buffer that decodes uploaded image bytes (via OpenCV `cv2.imdecode`) and retains the most recent frames in memory. Oldest frames are dropped once the buffer is full; dropped-frame and latency counters are tracked for telemetry.
- **`app/detector.py` — `LocalSpatialEvaluator`**: computes a **drift score** between the current frame and the previous buffered frame using grayscale conversion, absolute pixel difference (`cv2.absdiff`), and a fixed binary threshold (`cv2.threshold`). The drift score is the fraction of pixels that changed. If the frame resolution changed, the baseline is resized to match before comparison. A frame is classified `ANOMALY_TRIGGERED` if the score exceeds `DRIFT_THRESHOLD`, otherwise `NOMINAL_STABLE`.
- **`app/dispatch.py` — `CloudVLARouter`**: an async HTTP client (`httpx`) that posts to a single configurable cloud endpoint. Nominal frames get a small JSON metadata payload (no image bytes sent). Anomaly frames are re-encoded to JPEG and uploaded as multipart form data.

This is a straightforward, working frame-diff drift detector — not object or scene-level perception. It is a solid triage layer for deciding *which* frames are worth sending onward for richer analysis, not a replacement for that analysis.

---

## API

### `POST /v1/spatial/ingest`

Accepts a single image file upload (`multipart/form-data`, field name `file`). On each call the service:

1. Reads the current latest buffered frame as the baseline (if any).
2. Decodes and stores the new frame in the ring buffer.
3. Computes a drift score against the baseline (`0.0` if this is the first frame).
4. Dispatches the result to the configured cloud endpoint (metadata-only for nominal frames, full JPEG for anomalies).

Returns JSON with the frame ID, ingest timestamp, drift score, and the dispatch response.

### `GET /v1/spatial/telemetry`

Returns running counters: total frames processed, cloud uploads skipped (nominal frames), anomaly incidents triggered, failed frames, and buffer/latency metrics (dropped frames, retained frames, last ingest/pipeline/dispatch latency).

---

## Configuration

Settings are environment-backed (via `pydantic-settings`, loaded from the process environment or a local `.env` file):

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DRIFT_THRESHOLD` | `0.15` | Fraction of changed pixels (0–1) above which a frame is classified `ANOMALY_TRIGGERED`. |
| `TARGET_FRAME_RATE` | `30` | Target camera ingestion rate in frames per second (informational; not currently enforced by the service). |
| `CLOUD_VLA_ENDPOINT_URL` | `http://127.0.0.1:9000/v1/vla/commands` | The cloud endpoint that receives nominal-frame metadata and anomaly-frame JPEG uploads. |
| `MAX_BUFFER_FRAMES` | `90` | Maximum number of decoded frames retained in the ring buffer. |
| `EDGE_COMPRESSION_QUALITY` | `80` | JPEG quality (1–100) used when re-encoding anomaly frames for upload. |

## Running locally

```bash
# install
pip install -e ".[dev]"

# run the API
uvicorn app.main:app --reload

# run tests / lint / type-check
pytest
ruff check .
mypy .
```

Dependencies are OpenCV (`opencv-python-headless`), FastAPI, httpx, and pydantic — see `pyproject.toml`. There is no GStreamer, YOLO, or other model-inference dependency in this project.

---

## Roadmap / Not Yet Implemented

The following are real, intended directions for this project — none of them are built yet:

- **Actual object/scene-level perception at the edge.** Today's detector is a global pixel-diff drift score; it has no notion of objects, classes, or bounding boxes. A local model (e.g. a lightweight YOLO or RT-DETR variant) would let the edge triage on *what* changed, not just *how much*.
- **Temporal scene graphs.** Tracking entities and their relationships across frames is not implemented. Drift is computed frame-to-frame with no persistent state beyond the ring buffer.
- **GStreamer / zero-copy ingestion pipeline.** Frames currently arrive as HTTP multipart uploads and are decoded with OpenCV; there is no RTSP/GStreamer capture path or shared-memory/zero-copy frame handoff.
- **Multi-endpoint / adaptive routing.** Dispatch currently targets a single configured cloud endpoint; routing to multiple destinations based on frame content is not implemented.
- **Frame-rate-aware scheduling.** `TARGET_FRAME_RATE` is stored as configuration but not currently used to pace or throttle ingestion.
