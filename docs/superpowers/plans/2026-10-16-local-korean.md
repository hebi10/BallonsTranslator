# Local Korean Translator Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans. Scope is already authorized by the user's instruction to proceed in order.

**Goal:** A single recommended local Japanese→Korean workflow with installation, recovery and Windows launch support.

**Architecture:** Keep the existing Qt editor, lazy registries and LLM translation contracts. Add a local policy and native loopback Ollama transport, retain pipeline completion bits, and layer a Korean workspace over the editor.

**Tech Stack:** Python, Qt, upstream image modules, stdlib HTTP, Ollama.

**Spec:** ../specs/2026-10-16-local-korean-design.md

## Global Constraints

- No new runtime dependencies beyond existing selected backend requirements.
- Preserve source images, old config, downloads and manual text.
- No automatic cloud fallback or public upstream update in the local edition.
- Long operations run outside Qt's GUI thread.
- Windows and real-weight validation must be reported separately from unit tests.

## Review Focus

- Redirect/proxy cannot move inference away from loopback.
- Interrupted download or translation cannot become a successful page.
- Continue cannot erase manual translations or rerun already completed OCR.
- GUI queue cannot switch projects before all workers/save operations finish.
- Missing or malformed state cannot overwrite a user's existing project.

### Task 1: Local policy and transport

Files: utils/local_edition.py, utils/local_ollama.py, launch.py, modules/lazy_registry.py, modules/llm_chat.py; tests/test_local_edition.py, tests/test_local_ollama.py.

Interfaces: is_local_edition()->bool; activate_local_edition()->None; apply_local_profile(config)->None; local_readiness()->list[tuple[str,bool,str]]; prepare_local_models(progress,stop_event)->None. OllamaClient.status()->dict; pull(progress,stop_event)->None; test_translation()->str; chat(api_args,stop_event)->dict.

- [x] Write failing tests for fixed selection, data preservation and loopback-only structured response.
- [x] Run targeted pytest and confirm expected failures.
- [x] Implement native Ollama calls with no proxies/redirects, exact local tag, cancellable stream and timeouts.
- [x] Wire launch policy and registry selection, isolated config and native LLM dispatch.
- [x] Run targeted tests and commit.

### Task 2: Pipeline recovery

Files: ui/module_manager.py, utils/proj_imgtrans.py, tests/test_module_run_errors.py and resume tests.

Interfaces: ModuleManager.runImgtransPipeline(..., continue_mode=False); ProjImgTrans.failed_pages()->list[str].

- [x] Reproduce failed translation/inpaint wrongly marked complete and Continue repeating completed stages.
- [x] Gate stages by existing completion bits, preserve failures and stop safely.
- [x] Verify failed-stage recovery and manual translation preservation; commit.

### Task 3: Local workspace

Files: ui/local_workspace.py, ui/mainwindow.py, ui/configpanel.py, ui/io_thread.py; tests/test_local_workspace.py.

- [x] Test persisted queue validation and completion only after worker/save shutdown.
- [x] Add Korean readiness UI, progress/cancel workers, folder queue and editor entry.
- [x] Integrate safe stop snapshot, resume flags, failure signals and remove cloud setup controls from local UI.
- [x] Run offscreen smoke/regression tests and commit.

### Task 4: Windows launch package

Files: scripts/install_local.ps1, scripts/start_local.ps1, scripts/build_local_package.py, launch_local.bat, doc/LOCAL_WINDOWS.md, .github/workflows/local-windows.yml.

- [x] Implement isolated embedded runtime and dedicated Ollama loopback instance using official signed downloads.
- [x] Package tracked sources without credentials/models or local environments.
- [x] Validate archive and scripts where environment permits; document untested Windows behavior.

### Task 5: Integration and review

- [x] Run relevant tests, full suite, py_compile, git diff --check, UI render.
- [x] Independently review changes, fix significant issues and rerun affected tests.
- [ ] Push a dedicated branch and draft PR using GitHub connector; provide ZIP and exact setup instructions.

