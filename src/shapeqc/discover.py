"""Find subject folders with usable ENIGMA Shape .raw files under a user root."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from shapeqc.io import is_subject_dir

_LAYOUT_CSV = Path(__file__).resolve().parents[2] / "configs" / "site_layout.csv"

SKIP_DIR_NAMES = {
    "QA",
    "Freesurfer",
    "Indiana_FS",
    "UPenn_FS",
    "NEW_AutoQA_prep",
    "data",
    "scripts",
    "human_labels",
}
SKIP_PREFIXES = ("stats_", ".", "__")
MAX_DEPTH = 8


@dataclass
class DiscoverResult:
    subjects: list[Path]
    search_roots: list[Path]
    mode: str
    duplicates_dropped: list[str] = field(default_factory=list)
    skipped_dirs: list[str] = field(default_factory=list)


def load_site_layout(path: str | Path | None = None) -> dict[str, str]:
    csv_path = Path(path) if path else _LAYOUT_CSV
    out: dict[str, str] = {}
    if not csv_path.is_file():
        return out
    for line in csv_path.read_text(encoding="utf-8").splitlines()[1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        site, relpath = line.split(",", 1)
        out[site.strip()] = relpath.strip()
    return out


def labeled_sites() -> tuple[str, ...]:
    return tuple(load_site_layout())


def _skip_dir(name: str) -> bool:
    if name in SKIP_DIR_NAMES:
        return True
    if name.endswith("_FS") or name.endswith("_fs"):
        return True
    return any(name.startswith(p) for p in SKIP_PREFIXES)


def _walk_subjects(root: Path, *, max_depth: int = MAX_DEPTH) -> tuple[list[Path], list[str]]:
    found: list[Path] = []
    skipped: list[str] = []

    def rec(cur: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(p for p in cur.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError:
            return
        subject_children = [child for child in children if is_subject_dir(child)]
        found.extend(subject_children)
        for child in children:
            if is_subject_dir(child):
                continue
            if child.name == "bilat_shape" and subject_children:
                skipped.append(str(child))
                continue
            if _skip_dir(child.name):
                skipped.append(str(child))
                continue
            rec(child, depth + 1)

    if is_subject_dir(root):
        return [root], []
    rec(root, 0)
    return found, skipped


def _dedupe(subjects: list[Path], search_root: Path) -> tuple[list[Path], list[str]]:
    """Keep the shortest path per subject name inside one search root."""
    best: dict[str, Path] = {}
    dropped: list[str] = []
    for path in subjects:
        name = path.name
        prev = best.get(name)
        if prev is None:
            best[name] = path
            continue
        if len(path.parts) < len(prev.parts):
            dropped.append("{0}: drop {1} keep {2}".format(name, prev, path))
            best[name] = path
        else:
            dropped.append("{0}: drop {1} keep {2}".format(name, path, prev))
    ordered = sorted(best.values(), key=lambda p: str(p))
    return ordered, dropped


def search_roots_for(shape_root: str | Path) -> tuple[list[Path], str]:
    """Map a user path onto one or more folders that actually hold .raw files."""
    root = Path(shape_root).resolve()
    layout = load_site_layout()
    labeled = set(layout)
    try:
        children = {p.name for p in root.iterdir() if p.is_dir()}
    except OSError:
        children = set()

    if root.name == "ENIGMA_SZ" or len(children & labeled) >= 3:
        roots: list[Path] = []
        for site, rel in layout.items():
            candidate = root / rel
            if candidate.is_dir():
                roots.append(candidate)
            elif (root / site).is_dir():
                roots.append(root / site)
        return roots or [root], "enigma_sz"

    if root.name in layout:
        rel = Path(layout[root.name])
        parts = rel.parts
        if parts and parts[0] == root.name and len(parts) > 1:
            inner = root.joinpath(*parts[1:])
            if inner.is_dir():
                return [inner], "site"
        return [root], "site"

    return [root], "folder"


def discover_subjects(
    shape_root: str | Path,
    *,
    max_subjects: int | None = None,
) -> DiscoverResult:
    root = Path(shape_root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    roots, mode = search_roots_for(root)
    subjects: list[Path] = []
    skipped: list[str] = []
    dropped: list[str] = []
    cap = int(max_subjects) if max_subjects is not None and int(max_subjects) > 0 else None
    for search in roots:
        found, skip = _walk_subjects(search)
        kept, dup = _dedupe(found, search)
        skipped.extend(skip)
        dropped.extend(dup)
        for path in kept:
            subjects.append(path)
            if cap is not None and len(subjects) >= cap:
                return DiscoverResult(
                    subjects=subjects,
                    search_roots=roots,
                    mode=mode,
                    duplicates_dropped=dropped,
                    skipped_dirs=skipped,
                )

    return DiscoverResult(
        subjects=subjects,
        search_roots=roots,
        mode=mode,
        duplicates_dropped=dropped,
        skipped_dirs=skipped,
    )


def iter_discovered(
    shape_root: str | Path, *, max_subjects: int | None = None
) -> Iterable[Path]:
    result = discover_subjects(shape_root, max_subjects=max_subjects)
    if not result.subjects:
        raise SystemExit(
            "no subject folders with thick_*.raw under {0}".format(shape_root)
        )
    if result.duplicates_dropped:
        sys.stderr.write(
            "shapeqc: dropped {0} duplicate subject folder(s) (kept shortest path)\n".format(
                len(result.duplicates_dropped)
            )
        )
    if result.skipped_dirs:
        shown = ", ".join(Path(p).name for p in result.skipped_dirs[:12])
        extra = " ..." if len(result.skipped_dirs) > 12 else ""
        sys.stderr.write(
            "shapeqc: skipped {0} non-shape folder(s): {1}{2}\n".format(
                len(result.skipped_dirs), shown, extra
            )
        )
    yield from result.subjects
