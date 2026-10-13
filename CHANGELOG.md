# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2026-10-13

### Fixed
- Decompression bomb on `POST /v1/spatial/ingest`: a ~270 KB PNG declaring
  16000x16000 pixels passed the 10 MB upload cap, decoded to ~770 MB and was
  kept in the 90-frame buffer (server RSS 70 MB -> 840 MB from one request).
  Decoded frames are now capped by `MAX_FRAME_PIXELS` (default 8K UHD), enforced
  by OpenCV's own decode limit plus a post-decode check, and refused with
  `400` without being buffered.
- An image OpenCV refused to decode surfaced as `502 failed to push frame`
  instead of a `400` client error.

## [1.0.0] - 2026-10-13

First tagged release.

### Changed

- Shutdown cleanup of the cloud VLA client moved from the deprecated `@app.on_event("shutdown")` hook to a FastAPI lifespan handler (no behaviour change; removes the deprecation warning).

### Security

- Ingest race fixed: `AsyncFrameBuffer.push_frame` now returns `(previous, stored)` read atomically under the buffer lock, so concurrent uploads can no longer compute drift on another request's frame or upload the wrong image under a frame id.
- Upload size limit (`MAX_UPLOAD_BYTES`, default 10 MiB): oversized frames get `413` and are not read fully into memory.
- API-key gate (`API_KEY`, `X-API-Key` or Bearer) on ingest and telemetry; without a key only loopback clients are served unless `ALLOW_UNAUTHENTICATED=true`.

### Added

- `uv.lock` pinning the full dependency graph.
- Dependabot configuration for uv/pip dependencies and GitHub Actions (weekly).
- SECURITY.md describing private vulnerability reporting.
- This CHANGELOG.
