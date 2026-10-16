# Reader and export implementation plan

> **For agentic workers:** Use superpowers:executing-plans. Work inline; record actual verification below.

**Goal:** Read saved translations and export ordered vertical/horizontal composites without altering projects.
**Architecture:** Pure file/planning/export logic; isolated Qt reader and export dialogs; one integration widget owned by LocalWorkspace. No MainWindow rewrite.
**Tech Stack:** Existing qtpy/PyQt, Pillow, Python standard library.
**Spec:** docs/superpowers/specs/2026-10-16-reader-export-design.md

## Global Constraints
No new dependencies, no inference or network, no original overwrites. Default cross size 1080px; split length 12000px. Output caps 32000000 pixels and 60000px per side; source cap 40000000 pixels. Source-bootstrap distribution, not standalone EXE or mobile app.

## Review Focus
Corrupt project/history files remain unchanged. Missing/duplicate pages cannot silently disappear. Pixel exports preserve sequence, alpha and EXIF orientation. File changes after preview must fail export. Closing/cancelling dialogs must not delete live QThreads.

## Task 1: Pure reading/export service
- [x] Add tests/test_reader_export.py; verify missing implementation fails.
- [x] Add utils/reader_export.py: immutable Page/Book/options/plan, result inspection, page-boundary grouping, safe render/export, read position persistence.
- [x] Run tests including corrupt/missing/changed inputs, cancel, no overwrite, PNG/JPEG and horizontal/vertical pixels.

## Task 2: Reader and merge dialogs
- [x] Add tests/test_result_reader_ui.py for real Qt widgets and workers.
- [x] Add ui/result_reader.py: virtual reader canvas, bounded background image loader, navigation/history/close lifecycle.
- [x] Add ui/image_merge_dialog.py: ordered checklist, size/split controls, background previews and export with cancellation.
- [x] Verify Qt6 and Qt5 separately and capture offscreen screenshots.

## Task 3: Integrate and distribute
- [x] Add ui/reader_tools.py: read-only book loader, queue/dirty warnings, modal ownership and existing editor navigation.
- [x] Integrate tools in LocalWorkspace and extend existing regression runner and Qt5 CI scope.
- [x] Run original regressions, compile, diff checks and review new thread/error boundaries.
- [x] Commit on a separate feature branch, preserve default and existing release, deliver verified change-only update ZIP and patch and verification limits.

## Execution record
- Source baseline: codex/local-korean-translator at 30d89434d021ca2873b97c7ab5614ed5cee87495. Artifact PACKAGE_INFO revision is an upstream CI merge snapshot, not a feature commit.
- Local Qt packages are absent. Package installation is blocked by this container's DNS/network; pure tests run here, real Qt verification must use repository CI.

- Windows verification: 43 new tests pass with Qt6 and Qt5 separately; all 32 local regression files pass; seven original Qt5 regression files pass. Windows launcher and packaging tests pass with the existing symlink privilege skip.
- Offscreen screenshots use a Korean font loaded only in the CI temporary directory. No font files are distributed. Compact export settings scroll rather than overlap; workspace-owned dialogs preserve dark contrast.
- Resume ruling: commit application files separately from CI configuration. The prior run passed tests but could not push workflow changes with its runtime token; no token permission expansion is requested.
- GPU inference, user PC installation and phone gallery compatibility were not exercised.
