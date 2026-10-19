# Local automatic language and review workflow

Approved scope: apply the October 19 project feedback to the existing Windows
local edition. Keep the existing Python/Qt architecture, Qwen translation model,
CTD, manga_ocr, MIT48px and LaMa. No new product dependencies or cloud inference.

## Implementation

1. Base on dev `45686522536733c1d72500d985f537f25a71cfb3`; integrate the existing
   reader/export PR without replacing the newer English folder settings.
2. Default new folders to automatic English/Japanese OCR; retain manual old-job
   choices. Batch OCR by language, release child models between groups, preserve
   cancellation and block identity, and flag uncertain source text.
3. Persist block/page language overrides and work settings. Keep reading direction
   independent from language. Reuse explicitly selected work glossary and tone
   notes across folders without merging unrelated dialogue history.
4. Persist review issues and content-sensitive acknowledgments. Open the exact
   bubble, change source language explicitly, and preserve manual translations
   until the user requests rework. Retain the existing layout export guard.
5. Copy a selected image into an isolated preview project on a worker and run all
   four processing stages plus typesetting. Never overwrite the original project.
6. Include the integrated reader/export UI, a visible version, Windows CI on dev,
   and one source/bootstrap release rather than overlapping update ZIPs.

## Safety and verification

- Failed project loads restore the previous project identity and cannot save over
  malformed target JSON. Optional new metadata is normalized without passive writes.
- Review actions validate both project identity and the displayed content key.
- Preview completion waits for workers and image writes, saves partial progress on
  pause, restores prior stage settings, and does not advance the normal queue.
- Run the process-isolated local regression suite, relevant Qt5 checks, package
  boundary tests, and themed offscreen UI smoke checks before publishing.
- Automated tests use controlled OCR/model responses. Actual manga translation,
  GPU timing and restoration quality require representative user images.

## Execution notes

The user approved implementation of the preceding review. Runtime setup and
changes are isolated from the earlier checkout. Implementation was split into
language/pipeline, workspace, reader/export, review panel, preview worker and
integration owners; independent reviews cover the cross-feature call paths.
