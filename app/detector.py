"""Local spatial drift and structural anomaly evaluator."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import settings

logger = logging.getLogger(__name__)

FrameArray = NDArray[np.uint8]
FrameState = Literal["ANOMALY_TRIGGERED", "NOMINAL_STABLE"]


@dataclass(slots=True, frozen=True)
class DriftEvaluation:
    """Per-frame spatial drift result."""

    drift_score: float
    state: FrameState


class LocalSpatialEvaluator:
    """Lightweight local perception layer that detects structural scene drift."""

    async def compute_structural_drift(
        self,
        current_frame: FrameArray,
        baseline_frame: FrameArray,
    ) -> float:
        """Compute modified-pixel ratio between the current and baseline frames."""
        try:
            drift_score = await asyncio.to_thread(
                self._compute_structural_drift_sync,
                current_frame,
                baseline_frame,
            )
            state = self.classify_drift(drift_score)
            logger.info(
                "structural_drift_computed",
                extra={"drift_score": drift_score, "frame_state": state},
            )
            return drift_score
        except ValueError:
            logger.exception("structural_drift_validation_failed")
            raise
        except Exception as exc:
            logger.exception("structural_drift_computation_failed")
            raise RuntimeError("failed to compute structural drift") from exc

    def evaluate(self, drift_score: float) -> DriftEvaluation:
        """Return a complete drift evaluation record for a drift score."""
        return DriftEvaluation(drift_score=drift_score, state=self.classify_drift(drift_score))

    @staticmethod
    def classify_drift(drift_score: float) -> FrameState:
        """Classify a drift score against the configured anomaly threshold."""
        return "ANOMALY_TRIGGERED" if drift_score > settings.DRIFT_THRESHOLD else "NOMINAL_STABLE"

    @staticmethod
    def _compute_structural_drift_sync(
        current_frame: FrameArray,
        baseline_frame: FrameArray,
    ) -> float:
        if current_frame.size == 0 or baseline_frame.size == 0:
            raise ValueError("current_frame and baseline_frame must be non-empty arrays")
        if current_frame.shape[:2] != baseline_frame.shape[:2]:
            resized_baseline = cv2.resize(
                baseline_frame,
                (current_frame.shape[1], current_frame.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            baseline_frame = cast(FrameArray, resized_baseline)

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        baseline_gray = cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2GRAY)
        difference = cv2.absdiff(current_gray, baseline_gray)
        _, binary_delta = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
        changed_pixels = int(np.count_nonzero(binary_delta))
        total_pixels = int(binary_delta.shape[0] * binary_delta.shape[1])
        if total_pixels == 0:
            raise ValueError("frame resolution footprint cannot be zero")
        return float(changed_pixels / total_pixels)
