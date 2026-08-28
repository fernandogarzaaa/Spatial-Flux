"""Runtime settings for SpatialFlux."""

from __future__ import annotations

import logging

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Environment-backed configuration for edge perception and dispatch."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    DRIFT_THRESHOLD: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Spatial drift trigger threshold as a ratio between 0 and 1.",
    )
    TARGET_FRAME_RATE: int = Field(
        default=30,
        ge=1,
        le=240,
        description="Target camera ingestion rate in frames per second.",
    )
    CLOUD_VLA_ENDPOINT_URL: str = Field(
        default="http://127.0.0.1:9000/v1/vla/commands",
        description="Cloud endpoint receiving VLA telemetry and high-fidelity frame uploads.",
    )
    MAX_BUFFER_FRAMES: int = Field(
        default=90,
        ge=1,
        le=10000,
        description="Maximum number of decoded frames retained at the edge.",
    )
    EDGE_COMPRESSION_QUALITY: int = Field(
        default=80,
        ge=1,
        le=100,
        description="JPEG compression quality used for anomaly frame routing.",
    )
    MIN_DRIFT_REGION_AREA_RATIO: float = Field(
        default=0.002,
        ge=0.0,
        le=1.0,
        description="Minimum bounding-box area ratio for a drift region to be kept.",
    )
    MAX_DRIFT_REGIONS: int = Field(
        default=8,
        ge=1,
        le=100,
        description="Maximum number of drift regions returned per frame.",
    )
    DRIFT_REGION_DILATE_ITERATIONS: int = Field(
        default=2,
        ge=0,
        le=20,
        description="Dilation iterations applied to the binary delta before contour detection.",
    )

    @field_validator("CLOUD_VLA_ENDPOINT_URL")
    @classmethod
    def validate_cloud_endpoint(cls, value: str) -> str:
        """Validate endpoint syntax while keeping the stored value as a string."""
        try:
            HttpUrl(value)
        except ValueError as exc:
            logger.error("invalid_cloud_endpoint_url", extra={"endpoint": value})
            raise ValueError("CLOUD_VLA_ENDPOINT_URL must be a valid HTTP or HTTPS URL") from exc
        return value


settings = Settings()
