"""Build a source/bootstrap ZIP from tracked files, using current working contents."""
import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile


PRIVATE_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.btrans_cache', 'logs', 'tmp',
                'ballontrans_pylibs_win', 'local_runtime_win', 'local_ollama', 'result',
                'inpainted', 'mask'}
PRIVATE_NAMES = {'id_rsa', 'id_ed25519', 'credentials.json', 'token.json', 'tokens.json'}
PRIVATE_SUFFIXES = {'.pem', '.key', '.pfx', '.p12', '.pt', '.ckpt', '.onnx', '.gguf', '.safetensors'}


def package_allowed(name: str) -> bool:
    """Keep source and the public local preset, excluding runtime and personal data.

    >>> package_allowed('config/config.json')
    False
    >>> package_allowed('ballontranslator/launch.py')
    True
    """
    path = PurePosixPath(name)
    lower_parts = tuple(part.lower() for part in path.parts)
    if path.is_absolute() or '..' in path.parts or any(part in PRIVATE_DIRS for part in lower_parts):
        return False
    if path.name.lower().startswith('.env') or path.name.lower() in PRIVATE_NAMES:
        return False
    if path.suffix.lower() in PRIVATE_SUFFIXES:
        return False
    if lower_parts[:2] in (('data', 'models'), ('data', 'testpacks')):
        return False
    if lower_parts[:1] == ('config',) and name != 'config/hebi-local.json':
        return False
    return True


def build_package(root: Path, output: Path) -> int:
    root = root.resolve()
    names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode('utf-8').split('\0')
    selected = []
    for name in sorted(set(filter(None, names))):
        if not package_allowed(name):
            continue
        source = root / name
        if source.is_symlink() or (source.exists() and not source.resolve().is_relative_to(root)):
            raise ValueError(f'Symlinks are not included in release packages: {name}')
        if source.is_file():
            selected.append((name, source))
    if not selected:
        raise ValueError('No tracked source files were found. Add the intended source files to git first.')
    output.parent.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    metadata = {'format': 'source-bootstrap', 'standalone_executable': False,
                'revision': revision.stdout.strip() if revision.returncode == 0 else 'uncommitted',
                'contents': 'Current tracked working-tree files; runtime and model weights downloaded during setup.'}
    with ZipFile(output, 'x', ZIP_DEFLATED) as archive:
        for name, source in selected:
            archive.write(source, f'BallonsTranslator-Local/{name}')
        archive.writestr('BallonsTranslator-Local/PACKAGE_INFO.json', json.dumps(metadata, indent=2) + '\n')
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    count = build_package(args.root, args.output)
    print(f'Created {args.output} ({count} source files; no runtime or model weights).')


if __name__ == '__main__':
    main()
