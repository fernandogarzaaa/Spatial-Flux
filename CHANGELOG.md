# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
