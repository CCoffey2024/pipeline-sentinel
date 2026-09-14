# Changelog

All notable changes to Pipeline Sentinel are documented here. Versions use semantic versioning.

## [Unreleased]

### Added
- optional DINOv2 ViT-S/14 second-stage representation analysis after YOLO detection and tracking;
- bounded mature-track sampling controls for class filtering, cadence, crop padding, and maximum
  embeddings per frame;
- streamed representation evidence using a CSV row index, little-endian float32 vector file, and
  self-describing JSON manifest;
- per-run Operator Console controls, representation summary metrics, and an interactive results tab;
- aspect-preserving DINOv2 crop preparation and deterministic representation-stage tests.

### Changed
- the source-checkout Windows launcher installs the `dinov2` extra used by the production profile;
- DINOv2 representation evidence is independent from normal-reference anomaly scoring and does not
  assign event or alert meaning to visual change.

### Fixed
- mixed-resolution still-image sequences are now normalized to the first frame's canvas with
  aspect-preserving letterboxing before detection, tracking, annotation, and evidence-video writing;
- run manifests record the normalization policy, canonical canvas, observed source dimensions, and
  normalized-frame count.

## [0.12.0] - 2026-09-14

### Added
- unified operator media-ingest endpoint accepting one encoded video or one-or-more uploaded still-image frames;
- a single Sensor Ingest control surface for browser media uploads and read-in-place local folders/datasets;
- request-wide upload size enforcement for multi-image submissions;
- deterministic service tests covering unified video ingest, uploaded image sequences, mixed-media rejection, and the refactored operator controls;
- an explicit MVP release-readiness/acceptance checklist for external workstation testing;
- interactive completed-run results console with Overview, Detections, Tracks, Anomalies, Events / Alerts, and Downloads views;
- class summaries, detection-density visualization, run-storage reporting, and safe **Delete run & files** cleanup;
- codec-safe annotated-frame browser for evidence review when Chrome cannot decode the generated MP4 codec;
- codec-safe playback controls including Play/Pause, previous/next frame, speed controls, loop, scrubber, and timestamps;
- deterministic high-contrast per-class annotation colors with dark caption backing for easier human review;
- `TESTER-QUICKSTART.md` for packaged external workstation testing.

### Changed
- package version advanced to `0.12.0`;
- EO/IR/OTHER modality and sensor identity are now shared ingest metadata rather than being presented as video-specific controls;
- large image collections continue to use read-in-place local source adapters while small image sequences can be selected directly in the browser;
- the legacy `/api/jobs/run` encoded-video endpoint remains available for backward compatibility while the operator console uses `/api/jobs/media`;
- completed job detail views remain stable while background queue polling continues, preventing result-tab and playback resets;
- annotated review now uses stable class colors rather than gray/white track boxes and labels;
- Pipeline Sentinel itself is Apache-2.0 licensed while the Ultralytics YOLO provider is an explicit optional extra with separate upstream terms;
- the `operator` extra no longer installs Ultralytics automatically; external acceptance testing opts into `[operator,yolo]` explicitly;
- CI/release acceptance is hardened around the shipped operator package and Windows workstation target;
- v0.12.0 entered feature freeze after successful real-media acceptance runs; only release-blocking fixes and shipment documentation remain before tagging.

## [0.11.0] - 2026-09-13

### Added
- read-in-place local image-folder input in the operator console;
- native VisDrone sequence selection from existing local dataset roots without MP4 conversion;
- first-class UAVDT `UAV-benchmark-M` sequence discovery and frame streaming;
- operator APIs for local-source inspection and image-sequence run submission;
- local-source UI controls for sequence, modality, sensor ID, working FPS, frame step, and optional frame cap;
- deterministic tests for generic image folders, UAVDT discovery, local-source API behavior, and remote-mode filesystem protections.

### Changed
- package version advanced to `0.11.0`;
- local image datasets are treated as immutable external inputs and are not copied into the operator workspace;
- `PipelineSentinel.run_frames()` writes detection, track, anomaly, event, and alert CSV rows incrementally instead of retaining complete run tables in RAM;
- production orchestration now supports any lazy `FrameContext` stream with the same config/provenance lifecycle used by encoded video;
- local filesystem source APIs are automatically disabled when the operator service is bound beyond loopback.

## [0.10.0] - 2026-09-13

### Added
- local FastAPI operator service and bundled browser console;
- browser video upload with explicit sensor ID and EO/IR/OTHER modality selection;
- persistent JSON job registry with bounded worker execution and restart interruption handling;
- operator job APIs for status, alerts, events, artifacts, and multisensor fusion;
- one-click late-fusion workflow for completed sensor runs;
- `pipeline-sentinel-operator` launcher and Windows `start-operator.cmd` double-click helper;
- `operator` optional dependency extra for the UI/service plus YOLO runtime;
- local-service security guardrails, including loopback-only default binding and scoped artifact access;
- deterministic operator-job and API tests plus packaged UI/launcher release smoke checks.

### Changed
- package version advanced to `0.10.0`;
- application delivery is now separated from analytics: the UI and HTTP service consume the same production/fusion APIs and evidence artifacts as the CLI;
- the default operator worker count is one to avoid accidental concurrent GPU-heavy model stacks on a workstation.

## [0.9.0] - 2026-09-13

### Added
- config-driven semantic-event late fusion for two or more completed sensor runs;
- temporal consensus matching across distinct sensor IDs with configurable time tolerance;
- optional label agreement and explicit spatial-IoU gating for registered sensor products;
- `fusion_events.csv`, `fusion_alerts.csv`, `fusion_contributors.csv`, and `fusion_manifest.json` artifacts;
- strict fusion configuration with packaged and repository profiles;
- `validate-fusion-config` and `fuse-runs` CLI commands;
- deterministic tests for temporal, semantic, spatial, provenance, and alert-policy behavior.

### Changed
- package version advanced to `0.9.0`;
- EO/IR fusion happens after independent per-sensor inference and semantic event generation rather than mixing raw pixels or framework-specific model state;
- fused confidence is intentionally left unset because independent sensor/model confidences are not assumed to be calibrated onto one probability scale.

## [0.8.0] - 2026-09-13

### Added
- framework-neutral `Embedder` and anomaly-analysis contracts;
- optional DINOv2 representation adapter behind the embedder boundary;
- persisted normal-activity centroid/threshold reference artifacts;
- `pipeline-sentinel-fit-reference` reference-building command for curated normal crops;
- track-crop anomaly scoring with cosine distance from a normal centroid;
- `AnomalyObservation` runtime evidence contract and `anomalies.csv` run artifact;
- persistence-gated visual-anomaly event generation before alert policy;
- strict anomaly runtime configuration, disabled by default;
- deterministic tests for reference fitting, scoring, persistence, and end-to-end anomaly event promotion.

### Changed
- runtime flow now supports Detection -> Track -> AnomalyObservation -> Event -> Alert without treating a raw anomaly score as an alert;
- package version advanced to `0.8.0`;
- the DINOv2/Torch runtime remains optional so standard installs do not import or install the heavy representation backend.

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
