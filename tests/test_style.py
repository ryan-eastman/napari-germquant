"""Text style of the plugin (the engine's gate G7, applied here): no en or em dash anywhere, and plain ASCII in the
source, the README and the other docs, the CI workflow and the constraints files (the test data may carry a
micro sign, as the OME metadata of conftest does)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASCII_ONLY = ["README.md", "NAMING.md", "pyproject.toml", ".github/workflows/ci.yml", "constraints/*.txt",
              "src/**/*.py", "src/**/*.yaml", "docs/**/*.md"]
NO_DASHES = ASCII_ONLY + ["tests/*.py"]


def _files(patterns: list[str]) -> list[Path]:
    out: set[Path] = set()
    for pat in patterns:
        out.update(p for p in ROOT.glob(pat) if p.is_file())
    return sorted(out)


def test_no_en_or_em_dash_anywhere():
    files = _files(NO_DASHES)
    assert len(files) > 20
    bad = [str(p.relative_to(ROOT)) for p in files
           if chr(0x2013) in p.read_text(encoding="utf-8") or chr(0x2014) in p.read_text(encoding="utf-8")]
    assert not bad, f"en or em dash in: {bad}"


def test_source_docs_and_ci_files_are_ascii():
    bad = []
    for p in _files(ASCII_ONLY):
        try:
            p.read_bytes().decode("ascii")
        except UnicodeDecodeError as e:
            bad.append(f"{p.relative_to(ROOT)} (byte {e.start})")
    assert not bad, f"non-ASCII text in: {bad}"
