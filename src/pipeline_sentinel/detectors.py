from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from .types import Detection


class Detector(Protocol):
    """Framework-neutral detector contract."""

    name: str

    def detect(self, frame_number: int) -> list[Detection]: ...


class GroundTruthDetector:
    """Reference detector used only for deterministic integration tests."""

    name = "ground_truth"

    def __init__(self, annotation_path: Path) -> None:
        self.annotation_path = Path(annotation_path).resolve()
        if not self.annotation_path.exists():
            raise FileNotFoundError(self.annotation_path)
        self.annotations = pd.read_csv(self.annotation_path)
        required = {
            "frame_number",
            "label",
            "object_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "scenario_role",
        }
        missing = required - set(self.annotations.columns)
        if missing:
            raise ValueError(f"Ground truth missing required columns: {sorted(missing)}")

    def detect(self, frame_number: int) -> list[Detection]:
        rows = self.annotations.loc[self.annotations["frame_number"] == frame_number]
        return [
            Detection(
                frame_number=int(row.frame_number),
                label=str(row.label),
                confidence=1.0,
                x1=int(row.x1),
                y1=int(row.y1),
                x2=int(row.x2),
                y2=int(row.y2),
                source=self.name,
                object_id=int(row.object_id),
                scenario_role=str(row.scenario_role),
            )
            for row in rows.itertuples(index=False)
        ]
