"""
Regenerate the "dev fork" of the HA Power Predictor integration.

Copies the integration package and renames its domain so the fork can be
installed alongside the released version for side-by-side A/B testing. The fork
differs from the source only by its domain / name identifiers — everything else
(model code, defaults, translations) tracks the source verbatim. Stdlib-only;
it does NOT import the integration's modules or add the package to sys.path.

Usage:
    python tools/build_dev_fork.py            # rebuild dist/ha_power_predictor_dev in place
    python tools/build_dev_fork.py --check    # verify the committed fork is up to date
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
import tempfile
from pathlib import Path

# Dev-fork identifiers (the ONLY differences from the source package).
DEV_DOMAIN = "ha_power_predictor_dev"
DEV_NAME = "HA Power Predictor (Dev)"
DEV_INTEGRATION_NAME = "Power Predictor Dev"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _REPO_ROOT / "custom_components" / "ha_power_predictor"
_DEST = _REPO_ROOT / "dist" / "ha_power_predictor_dev"

# Exact string replacements, grouped by the file (relative to the package root)
# they apply to. Each is (old, new); the old string must appear exactly once.
_REPLACEMENTS: dict[str, list[tuple[str, str]]] = {
    "manifest.json": [
        ('"domain": "ha_power_predictor"', f'"domain": "{DEV_DOMAIN}"'),
        ('"name": "HA Power Predictor"', f'"name": "{DEV_NAME}"'),
    ],
    "const.py": [
        ('DOMAIN = "ha_power_predictor"', f'DOMAIN = "{DEV_DOMAIN}"'),
        (
            'DEFAULT_INTEGRATION_NAME = "Power Predictor"',
            f'DEFAULT_INTEGRATION_NAME = "{DEV_INTEGRATION_NAME}"',
        ),
    ],
}


def _copy_package(source: Path, dest: Path) -> None:
    """Replace `dest` with a fresh copy of `source`, dropping caches."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        source,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _apply_replacement(path: Path, old: str, new: str) -> None:
    """Replace `old` with `new` in `path`, failing loudly if absent/duplicated."""
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise ValueError(
            f"Expected exactly one occurrence of {old!r} in {path}, found {count}. "
            "The source format changed — update build_dev_fork.py."
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


def _rename_domain(dest: Path) -> list[str]:
    """Apply all configured replacements under `dest`; return a summary list."""
    applied: list[str] = []
    for rel_path, replacements in _REPLACEMENTS.items():
        target = dest / rel_path
        for old, new in replacements:
            _apply_replacement(target, old, new)
            applied.append(f"{rel_path}: {old}  ->  {new}")
    return applied


def _build(dest: Path) -> list[str]:
    """Copy the package to `dest` and rename its domain. Returns replacements."""
    _copy_package(_SOURCE, dest)
    return _rename_domain(dest)


def _count_files(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file())


def _trees_match(left: Path, right: Path) -> bool:
    """Recursively compare two directory trees by content."""
    cmp = filecmp.dircmp(left, right)
    if cmp.left_only or cmp.right_only or cmp.diff_files or cmp.funny_files:
        return False
    return all(
        _trees_match(left / sub, right / sub) for sub in cmp.common_dirs
    )


def _run_check() -> int:
    """Regenerate into a temp dir and compare to the committed fork."""
    with tempfile.TemporaryDirectory() as tmp:
        candidate = Path(tmp) / "ha_power_predictor_dev"
        _build(candidate)
        if not _DEST.exists():
            print(f"FAIL: {_DEST} does not exist; run build_dev_fork.py to create it.")
            return 1
        if _trees_match(candidate, _DEST):
            print(f"OK: {_DEST} is in sync with the source package.")
            return 0
        print(f"FAIL: {_DEST} is stale; re-run build_dev_fork.py to regenerate it.")
        return 1


def _run_build() -> int:
    applied = _build(_DEST)
    n_files = _count_files(_DEST)
    print(f"Rebuilt dev fork at {_DEST}")
    print(f"Files written: {n_files}")
    print(f"Replacements applied ({len(applied)}):")
    for line in applied:
        print(f"  {line}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate into a temp dir and verify the committed fork matches",
    )
    args = parser.parse_args(argv)
    return _run_check() if args.check else _run_build()


if __name__ == "__main__":
    sys.exit(main())
