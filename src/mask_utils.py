from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


def polygon_to_mask(points: Sequence[Point], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 3:
        return mask
    poly = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [poly], 1)
    return mask


def polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(points: Sequence[Point]) -> Tuple[float, float]:
    if len(points) < 3:
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) == 0:
            return 0.0, 0.0
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    a = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    if abs(a) < 1e-8:
        return float(x.mean()), float(y.mean())
    factor = 1.0 / (3.0 * a)
    cx = factor * np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y))
    cy = factor * np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y))
    return float(cx), float(cy)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(a, b).sum()
    return float(inter / union)


def mask_area(mask: np.ndarray) -> int:
    return int(mask.astype(bool).sum())


def points_to_json(points: Sequence[Point]) -> str:
    return json.dumps([{"x": int(x), "y": int(y)} for x, y in points])


def points_from_json(value: str):
    items = json.loads(value)
    return [(int(item["x"]), int(item["y"])) for item in items]
