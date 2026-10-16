"""Run the local Windows edition's regression files in fresh Python processes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


TEST_FILES = (
    'tests/test_local_edition.py',
    'tests/test_local_ollama.py',
    'tests/test_local_dialogue_context.py',
    'tests/test_local_bubble_grouping.py',
    'tests/test_local_workspace.py',
    'tests/test_korean_segmentation.py',
    'tests/test_korean_layout.py',
    'tests/test_pipeline_resume.py',
    'tests/test_llm_translation_contract.py',
    'tests/test_llm_translation_integration.py',
    'tests/test_llm_translation_history.py',
    'tests/test_llm_translation_summary.py',
    'tests/test_llm_translation_memory.py',
    'tests/test_llm_vision.py',
    'tests/test_llm_chat.py',
    'tests/test_llm_profiles.py',
    'tests/test_module_run_errors.py',
    'tests/test_run_pipeline_dialog.py',
    'tests/test_image_io.py',
    'tests/test_canvas_save_state.py',
    'tests/test_launch_restart.py',
    'tests/test_core_requirements.py',
    'tests/test_torch_install_helper.py',
)


def main() -> int:
    # Windows CI can redirect stdout using a legacy code page. Preserve Korean
    # diagnostics instead of crashing while reporting the original test error.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--files', nargs='+', default=TEST_FILES)
    parser.add_argument('--timeout', type=int, default=300, help='Maximum seconds per test file')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    for name in args.files:
        if not (root / name).is_file():
            parser.error(f'Test file is missing: {name}')
    environment = dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    failures = []
    for name in args.files:
        print(f'RUN {name}', flush=True)
        try:
            # Qt bindings retain native state between test modules. Process isolation
            # makes each file independent and preserves crashes as nonzero failures.
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', '-q', name], cwd=root,
                env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', timeout=args.timeout,
            )
            print(result.stdout, end='', flush=True)
            if result.returncode:
                failures.append(name)
                print(f'FAIL {name} (exit {result.returncode})', flush=True)
            else:
                print(f'PASS {name}', flush=True)
        except subprocess.TimeoutExpired:
            failures.append(name)
            print(f'FAIL {name} (timeout after {args.timeout}s)', flush=True)
    if failures:
        print(f'Failed {len(failures)} of {len(args.files)} files: {", ".join(failures)}', flush=True)
        return 1
    print(f'Passed all {len(args.files)} regression files.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
