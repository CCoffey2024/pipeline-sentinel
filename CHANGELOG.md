# Changelog

All notable changes to Pipeline Sentinel are documented here. Versions use semantic versioning.

## [0.7.0] - 2026-09-13

### Added
- tag-driven GitHub Release workflow;
- wheel and source-distribution release artifacts;
- SHA-256 checksum manifest for release downloads;
- clean-environment wheel smoke test before release creation;
- `python -m pipeline_sentinel` module entry point;
- release/version preflight checks;
- short-lived installable distribution artifacts from normal CI;
- release and rollback documentation.

### Changed
- package version now has one source of truth in `pipeline_sentinel.__version__`, consumed by Hatch;
- version advanced to `0.7.0`;
- release candidates must pass lint, tests, packaging, install, CLI, and bundled-config gates before a GitHub Release is created.

## [0.6.0] - 2026-09-13

### Added
- config-driven production `run` command;
- strict typed YAML validation;
- bundled production profile;
- run IDs, effective-config snapshots, structured JSONL lifecycle logging, and run status artifacts;
- package build and clean-wheel smoke testing in CI.

## [0.5.0] - 2026-09-13

### Added
- framework-neutral tracking, event, and alert contracts;
- deterministic IoU tracker;
- scenario-role and baseline dwell event detectors;
- severity-based alert policy;
- `tracks.csv` and `events.csv` runtime artifacts.

## [0.4.0] - 2026-09-13

### Added
- shared-class VisDrone/COCO fixed-IoU evaluator;
- per-class TP/FP/FN, precision, recall, F1, and matched-IoU evidence.

## [0.3.0] - 2026-09-13

### Added
- first-class VisDrone2019-VID image-sequence ingestion and run support;
- generic `FrameContext` stream execution;
- scoped ground-truth export for processed frames.

## [0.2.0] - 2026-09-13

### Added
- optional Ultralytics YOLO adapter behind the detector contract;
- image-bearing `FrameContext` runtime contract;
- YOLO CLI execution path.

## [0.1.0] - 2026-09-12

### Added
- initial installable application foundation;
- canonical ingest/data contracts;
- deterministic synthetic/reference pipeline;
- CLI, tests, and continuous integration.
