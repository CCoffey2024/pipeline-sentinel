from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

SHARED_ONTOLOGY_NAME = "pipeline-sentinel-visdrone-shared-v1"
SHARED_CLASSES = ("person", "bicycle", "motor", "car", "truck", "bus")

PREDICTION_TO_SHARED = {
    "person": "person",
    "bicycle": "bicycle",
    "motorcycle": "motor",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
}

VISDRONE_TO_SHARED = {
    "pedestrian": "person",
    "people": "person",
    "bicycle": "bicycle",
    "motor": "motor",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
}

PREDICTION_REQUIRED_COLUMNS = {
    "frame_number",
    "label",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
}
GROUND_TRUTH_REQUIRED_COLUMNS = {
    "frame_index",
    "label",
    "ignored",
    "x1",
    "y1",
    "x2",
    "y2",
}

MATCH_COLUMNS = [
    "frame_number",
    "shared_class",
    "status",
    "prediction_index",
    "prediction_label",
    "prediction_confidence",
    "pred_x1",
    "pred_y1",
    "pred_x2",
    "pred_y2",
    "ground_truth_index",
    "ground_truth_label",
    "ground_truth_target_id",
    "gt_x1",
    "gt_y1",
    "gt_x2",
    "gt_y2",
    "iou",
]


@dataclass(slots=True)
class EvaluationResult:
    summary: dict[str, Any]
    class_metrics: pd.DataFrame
    matches: pd.DataFrame
    excluded_predictions: pd.DataFrame
    excluded_ground_truth: pd.DataFrame


@dataclass(frozen=True, slots=True)
class BenchmarkArtifacts:
    summary_json: Path
    class_metrics_csv: Path
    matches_csv: Path
    excluded_predictions_csv: Path
    excluded_ground_truth_csv: Path


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Compute intersection over union for two XYXY boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _validate_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def _as_numeric(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    invalid = frame.loc[(frame["x2"] <= frame["x1"]) | (frame["y2"] <= frame["y1"])]
    if not invalid.empty:
        row_preview = ", ".join(str(index) for index in invalid.index[:5])
        raise ValueError(f"{name} contains invalid XYXY boxes at row(s): {row_preview}")


def _box(row: Any, prefix: str = "") -> tuple[float, float, float, float]:
    return (
        float(getattr(row, f"{prefix}x1")),
        float(getattr(row, f"{prefix}y1")),
        float(getattr(row, f"{prefix}x2")),
        float(getattr(row, f"{prefix}y2")),
    )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _counts_by_label(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    counts = frame["label"].astype(str).value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _prepare_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(predictions, PREDICTION_REQUIRED_COLUMNS, "detections.csv")
    prepared = predictions.copy().reset_index(drop=True)
    prepared["prediction_index"] = prepared.index.astype(int)
    prepared["frame_number"] = pd.to_numeric(prepared["frame_number"], errors="raise").astype(int)
    prepared["confidence"] = pd.to_numeric(prepared["confidence"], errors="raise")
    _as_numeric(prepared, ["x1", "y1", "x2", "y2"], "detections.csv")
    prepared["label"] = prepared["label"].astype(str)
    prepared["shared_class"] = prepared["label"].map(PREDICTION_TO_SHARED)
    return prepared


def _prepare_ground_truth(ground_truth: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(ground_truth, GROUND_TRUTH_REQUIRED_COLUMNS, "ground_truth.csv")
    prepared = ground_truth.copy().reset_index(drop=True)
    prepared["ground_truth_index"] = prepared.index.astype(int)
    prepared["frame_number"] = pd.to_numeric(prepared["frame_index"], errors="raise").astype(int)
    _as_numeric(prepared, ["x1", "y1", "x2", "y2"], "ground_truth.csv")
    prepared["ignored_bool"] = _boolean_series(prepared["ignored"])
    prepared["label"] = prepared["label"].astype(str)
    prepared["shared_class"] = prepared["label"].map(VISDRONE_TO_SHARED)
    return prepared


def _match_frame_class(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    frame_number: int,
    shared_class: str,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    prediction_rows = predictions.loc[
        (predictions["frame_number"] == frame_number)
        & (predictions["shared_class"] == shared_class)
    ].sort_values(["confidence", "prediction_index"], ascending=[False, True])
    ground_truth_rows = ground_truth.loc[
        (ground_truth["frame_number"] == frame_number)
        & (ground_truth["shared_class"] == shared_class)
    ].sort_values("ground_truth_index")

    gt_records = {int(row.ground_truth_index): row for row in ground_truth_rows.itertuples(index=False)}
    unmatched_gt = set(gt_records)
    rows: list[dict[str, Any]] = []

    for prediction in prediction_rows.itertuples(index=False):
        best_gt_index: int | None = None
        best_iou = 0.0
        for gt_index in sorted(unmatched_gt):
            candidate = gt_records[gt_index]
            candidate_iou = box_iou(_box(prediction), _box(candidate))
            if candidate_iou > best_iou:
                best_iou = candidate_iou
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= iou_threshold:
            matched_gt = gt_records[best_gt_index]
            unmatched_gt.remove(best_gt_index)
            status = "TP"
        else:
            matched_gt = gt_records.get(best_gt_index) if best_gt_index is not None else None
            status = "FP"

        rows.append(
            {
                "frame_number": frame_number,
                "shared_class": shared_class,
                "status": status,
                "prediction_index": int(prediction.prediction_index),
                "prediction_label": str(prediction.label),
                "prediction_confidence": float(prediction.confidence),
                "pred_x1": float(prediction.x1),
                "pred_y1": float(prediction.y1),
                "pred_x2": float(prediction.x2),
                "pred_y2": float(prediction.y2),
                "ground_truth_index": (
                    int(matched_gt.ground_truth_index) if matched_gt is not None else None
                ),
                "ground_truth_label": str(matched_gt.label) if matched_gt is not None else None,
                "ground_truth_target_id": (
                    int(matched_gt.target_id)
                    if matched_gt is not None and hasattr(matched_gt, "target_id")
                    else None
                ),
                "gt_x1": float(matched_gt.x1) if matched_gt is not None else None,
                "gt_y1": float(matched_gt.y1) if matched_gt is not None else None,
                "gt_x2": float(matched_gt.x2) if matched_gt is not None else None,
                "gt_y2": float(matched_gt.y2) if matched_gt is not None else None,
                "iou": float(best_iou),
            }
        )

    for gt_index in sorted(unmatched_gt):
        missed_gt = gt_records[gt_index]
        rows.append(
            {
                "frame_number": frame_number,
                "shared_class": shared_class,
                "status": "FN",
                "prediction_index": None,
                "prediction_label": None,
                "prediction_confidence": None,
                "pred_x1": None,
                "pred_y1": None,
                "pred_x2": None,
                "pred_y2": None,
                "ground_truth_index": int(missed_gt.ground_truth_index),
                "ground_truth_label": str(missed_gt.label),
                "ground_truth_target_id": (
                    int(missed_gt.target_id) if hasattr(missed_gt, "target_id") else None
                ),
                "gt_x1": float(missed_gt.x1),
                "gt_y1": float(missed_gt.y1),
                "gt_x2": float(missed_gt.x2),
                "gt_y2": float(missed_gt.y2),
                "iou": None,
            }
        )

    return rows


def evaluate_visdrone_tables(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    iou_threshold: float = 0.50,
) -> EvaluationResult:
    """Evaluate prediction and VisDrone tables in the explicit shared ontology.

    This is a deterministic fixed-IoU Pipeline Sentinel benchmark, not the official VisDrone AP
    implementation. It intentionally keeps provider/native labels intact and applies ontology
    mapping only in the evaluation layer.
    """

    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")

    prepared_predictions = _prepare_predictions(predictions)
    prepared_ground_truth = _prepare_ground_truth(ground_truth)

    evaluated_predictions = prepared_predictions.loc[
        prepared_predictions["shared_class"].notna()
    ].copy()
    excluded_predictions = prepared_predictions.loc[
        prepared_predictions["shared_class"].isna()
    ].copy()

    ignored_ground_truth = prepared_ground_truth.loc[prepared_ground_truth["ignored_bool"]].copy()
    nonignored_ground_truth = prepared_ground_truth.loc[~prepared_ground_truth["ignored_bool"]].copy()
    evaluated_ground_truth = nonignored_ground_truth.loc[
        nonignored_ground_truth["shared_class"].notna()
    ].copy()
    excluded_ground_truth = nonignored_ground_truth.loc[
        nonignored_ground_truth["shared_class"].isna()
    ].copy()

    frame_numbers = sorted(
        set(evaluated_predictions["frame_number"].astype(int))
        | set(evaluated_ground_truth["frame_number"].astype(int))
    )

    match_rows: list[dict[str, Any]] = []
    for shared_class in SHARED_CLASSES:
        class_frames = sorted(
            set(
                evaluated_predictions.loc[
                    evaluated_predictions["shared_class"] == shared_class, "frame_number"
                ].astype(int)
            )
            | set(
                evaluated_ground_truth.loc[
                    evaluated_ground_truth["shared_class"] == shared_class, "frame_number"
                ].astype(int)
            )
        )
        for frame_number in class_frames:
            match_rows.extend(
                _match_frame_class(
                    evaluated_predictions,
                    evaluated_ground_truth,
                    frame_number=frame_number,
                    shared_class=shared_class,
                    iou_threshold=iou_threshold,
                )
            )

    matches = pd.DataFrame(match_rows, columns=MATCH_COLUMNS)
    metrics_rows: list[dict[str, Any]] = []
    for shared_class in SHARED_CLASSES:
        class_matches = matches.loc[matches["shared_class"] == shared_class]
        prediction_count = int(
            (evaluated_predictions["shared_class"] == shared_class).sum()
        )
        ground_truth_count = int(
            (evaluated_ground_truth["shared_class"] == shared_class).sum()
        )
        true_positive = int((class_matches["status"] == "TP").sum())
        false_positive = int((class_matches["status"] == "FP").sum())
        false_negative = int((class_matches["status"] == "FN").sum())
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        matched_ious = pd.to_numeric(
            class_matches.loc[class_matches["status"] == "TP", "iou"], errors="coerce"
        ).dropna()
        metrics_rows.append(
            {
                "shared_class": shared_class,
                "predictions": prediction_count,
                "ground_truth": ground_truth_count,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "mean_matched_iou": (
                    float(matched_ious.mean()) if not matched_ious.empty else None
                ),
            }
        )

    class_metrics = pd.DataFrame(metrics_rows)
    total_tp = int(class_metrics["true_positive"].sum())
    total_fp = int(class_metrics["false_positive"].sum())
    total_fn = int(class_metrics["false_negative"].sum())
    overall_precision = _safe_ratio(total_tp, total_tp + total_fp)
    overall_recall = _safe_ratio(total_tp, total_tp + total_fn)
    matched_ious = pd.to_numeric(
        matches.loc[matches["status"] == "TP", "iou"], errors="coerce"
    ).dropna()

    summary: dict[str, Any] = {
        "benchmark_name": "Pipeline Sentinel shared-class IoU benchmark",
        "benchmark_scope": "fixed-IoU detection precision/recall; not official VisDrone AP",
        "ontology": {
            "name": SHARED_ONTOLOGY_NAME,
            "shared_classes": list(SHARED_CLASSES),
            "prediction_to_shared": PREDICTION_TO_SHARED,
            "visdrone_to_shared": VISDRONE_TO_SHARED,
            "human_mapping_note": (
                "VisDrone pedestrian and people are intentionally collapsed into shared class person."
            ),
        },
        "iou_threshold": float(iou_threshold),
        "counts": {
            "predictions_total": int(len(prepared_predictions)),
            "predictions_evaluated": int(len(evaluated_predictions)),
            "predictions_out_of_ontology": int(len(excluded_predictions)),
            "ground_truth_total": int(len(prepared_ground_truth)),
            "ground_truth_ignored": int(len(ignored_ground_truth)),
            "ground_truth_evaluated": int(len(evaluated_ground_truth)),
            "ground_truth_out_of_ontology": int(len(excluded_ground_truth)),
            "frames_with_evaluable_objects_or_predictions": int(len(frame_numbers)),
            "first_evaluable_frame": int(frame_numbers[0]) if frame_numbers else None,
            "last_evaluable_frame": int(frame_numbers[-1]) if frame_numbers else None,
        },
        "overall": {
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": _f1(overall_precision, overall_recall),
            "mean_matched_iou": float(matched_ious.mean()) if not matched_ious.empty else None,
        },
        "excluded_prediction_labels": _counts_by_label(excluded_predictions),
        "excluded_ground_truth_labels": _counts_by_label(excluded_ground_truth),
        "ignored_ground_truth_labels": _counts_by_label(ignored_ground_truth),
        "limitations": [
            "This is not the official VisDrone AP/mAP evaluator.",
            "Ground-truth rows marked ignored are excluded from scoring; ignored-region overlap does not suppress predictions in this v1 benchmark.",
            "Predictions and non-ignored ground truth outside the shared ontology are reported but not scored.",
            "Metrics are fixed-threshold precision/recall/F1 at one IoU threshold, not confidence-swept AP.",
        ],
    }

    return EvaluationResult(
        summary=summary,
        class_metrics=class_metrics,
        matches=matches,
        excluded_predictions=excluded_predictions,
        excluded_ground_truth=excluded_ground_truth,
    )


def evaluate_visdrone_run(
    run_dir: Path,
    *,
    iou_threshold: float = 0.50,
    output_dir: Path | None = None,
) -> tuple[EvaluationResult, BenchmarkArtifacts]:
    """Evaluate one saved ``run-visdrone`` artifact directory and persist benchmark outputs."""

    run_dir = Path(run_dir).expanduser().resolve()
    detections_path = run_dir / "detections.csv"
    ground_truth_path = run_dir / "ground_truth.csv"
    manifest_path = run_dir / "run_manifest.json"
    for required_path in (detections_path, ground_truth_path, manifest_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest.get("metadata") or {}
    if metadata.get("ground_truth_scope") != "processed_frames":
        raise ValueError(
            "run_manifest.json does not declare ground_truth_scope='processed_frames'. "
            "Re-run 'pipeline-sentinel run-visdrone' before benchmarking so predictions and ground truth use the same frame window."
        )

    result = evaluate_visdrone_tables(
        pd.read_csv(detections_path),
        pd.read_csv(ground_truth_path),
        iou_threshold=iou_threshold,
    )
    result.summary["run"] = {
        "run_dir": str(run_dir),
        "pipeline_sentinel_version": manifest.get("pipeline_sentinel_version"),
        "input_source": manifest.get("input_source"),
        "detector_backend": manifest.get("detector_backend"),
        "frames_processed": manifest.get("frames_processed"),
        "first_frame_number": manifest.get("first_frame_number"),
        "last_frame_number": manifest.get("last_frame_number"),
        "split": metadata.get("split"),
        "sequence_id": metadata.get("sequence_id"),
        "model": metadata.get("detector_model"),
        "confidence_threshold": metadata.get("detector_confidence"),
        "nms_iou_threshold": metadata.get("detector_nms_iou"),
        "image_size": metadata.get("detector_imgsz"),
        "device": metadata.get("detector_device"),
        "class_ids": metadata.get("detector_class_ids"),
    }

    output_dir = (
        Path(output_dir).expanduser().resolve() if output_dir is not None else run_dir / "benchmark"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = BenchmarkArtifacts(
        summary_json=output_dir / "benchmark_summary.json",
        class_metrics_csv=output_dir / "class_metrics.csv",
        matches_csv=output_dir / "matches.csv",
        excluded_predictions_csv=output_dir / "excluded_predictions.csv",
        excluded_ground_truth_csv=output_dir / "excluded_ground_truth.csv",
    )
    artifacts.summary_json.write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result.class_metrics.to_csv(artifacts.class_metrics_csv, index=False)
    result.matches.to_csv(artifacts.matches_csv, index=False)
    result.excluded_predictions.to_csv(artifacts.excluded_predictions_csv, index=False)
    result.excluded_ground_truth.to_csv(artifacts.excluded_ground_truth_csv, index=False)
    return result, artifacts
