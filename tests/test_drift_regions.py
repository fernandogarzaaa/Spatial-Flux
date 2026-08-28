"""Tests for region-level motion localization in LocalSpatialEvaluator."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import settings
from app.detector import LocalSpatialEvaluator


def _blank_frame(size: int = 256) -> np.ndarray:
    return np.full((size, size, 3), 20, dtype=np.uint8)


@pytest.mark.asyncio
async def test_single_rectangle_produces_one_matching_region() -> None:
    evaluator = LocalSpatialEvaluator()
    baseline = _blank_frame()
    current = baseline.copy()
    cv2.rectangle(current, (40, 40), (100, 100), (255, 255, 255), thickness=-1)

    regions = await evaluator.compute_drift_regions(current, baseline)

    assert len(regions) == 1
    region = regions[0]
    # Dilation grows the blob, so allow tolerance around the drawn rectangle.
    assert 30 <= region.x <= 45
    assert 30 <= region.y <= 45
    assert 45 <= region.width <= 80
    assert 45 <= region.height <= 80
    assert region.area_ratio > 0.0


@pytest.mark.asyncio
async def test_two_separated_regions_returned_sorted_largest_first() -> None:
    evaluator = LocalSpatialEvaluator()
    baseline = _blank_frame(size=300)
    current = baseline.copy()
    # Large region, top-left.
    cv2.rectangle(current, (10, 10), (70, 70), (255, 255, 255), thickness=-1)
    # Small region, bottom-right, well separated from the first.
    cv2.rectangle(current, (220, 220), (240, 240), (255, 255, 255), thickness=-1)

    regions = await evaluator.compute_drift_regions(current, baseline)

    assert len(regions) == 2
    assert regions[0].area_ratio >= regions[1].area_ratio
    # The larger drawn rectangle should correspond to the first (largest) region.
    assert regions[0].width * regions[0].height > regions[1].width * regions[1].height


@pytest.mark.asyncio
async def test_tiny_region_below_min_area_threshold_is_filtered_out() -> None:
    evaluator = LocalSpatialEvaluator()
    baseline = _blank_frame(size=400)
    current = baseline.copy()
    # A 3x3 speck is far below the default MIN_DRIFT_REGION_AREA_RATIO for a 400x400 frame.
    cv2.rectangle(current, (200, 200), (202, 202), (255, 255, 255), thickness=-1)

    regions = await evaluator.compute_drift_regions(current, baseline)

    assert regions == ()


@pytest.mark.asyncio
async def test_region_count_capped_at_max_drift_regions() -> None:
    evaluator = LocalSpatialEvaluator()
    baseline = _blank_frame(size=600)
    current = baseline.copy()

    # Draw more separated qualifying blobs than MAX_DRIFT_REGIONS allows.
    blob_count = settings.MAX_DRIFT_REGIONS + 4
    for i in range(blob_count):
        row = i // 4
        col = i % 4
        top_left = (20 + col * 140, 20 + row * 140)
        bottom_right = (top_left[0] + 40, top_left[1] + 40)
        cv2.rectangle(current, top_left, bottom_right, (255, 255, 255), thickness=-1)

    regions = await evaluator.compute_drift_regions(current, baseline)

    assert len(regions) == settings.MAX_DRIFT_REGIONS


@pytest.mark.asyncio
async def test_compute_drift_matches_scalar_and_region_apis() -> None:
    evaluator = LocalSpatialEvaluator()
    baseline = _blank_frame()
    current = baseline.copy()
    cv2.rectangle(current, (40, 40), (100, 100), (255, 255, 255), thickness=-1)

    scalar_score = await evaluator.compute_structural_drift(current, baseline)
    regions = await evaluator.compute_drift_regions(current, baseline)
    evaluation = await evaluator.compute_drift(current, baseline)

    assert evaluation.drift_score == pytest.approx(scalar_score)
    assert evaluation.regions == regions
