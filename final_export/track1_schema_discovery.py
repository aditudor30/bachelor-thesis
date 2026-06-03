"""Local repository discovery for possible Track 1 schema references."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


DEFAULT_TRACK1_SEARCH_TERMS = [
    "track1.txt",
    "track1_schema",
    "Track 1",
    "track 1",
    "submission",
    "AICity",
    "AI City",
    "PhysicalAI",
    "SmartSpaces",
    "output format",
    "evaluation format",
    "challenge format",
]

TEXT_EXTENSIONS = set([".md", ".txt", ".yaml", ".yml", ".json", ".py", ".csv"])
SKIP_DIRS = set([".git", "__pycache__", ".pytest_cache", ".mypy_cache"])


def discover_track1_schema(
    repo_root: Union[str, Path],
    search_terms: Optional[List[str]] = None,
    max_file_size_mb: int = 5,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Search local repository text files for possible Track 1 schema references."""
    root = Path(repo_root)
    terms = search_terms if search_terms is not None else DEFAULT_TRACK1_SEARCH_TERMS
    files = _collect_text_files(root, max_file_size_mb=max_file_size_mb)
    matches = []
    candidate_schema_files = set()
    for path in _progress_iter(files, show_progress, "discover Track 1 schema", "file"):
        for item in _scan_file_for_terms(path, root, terms):
            matches.append(item)
            if _looks_like_confirmed_schema_line(item["line_text"]):
                candidate_schema_files.add(item["file_path"])
    found = len(candidate_schema_files) > 0
    notes = []
    if found:
        notes.append("Possible local schema fragments were found. Review candidate_schema_files manually.")
    else:
        notes.append(
            "Official Track 1 schema was not found in the local repository. "
            "Do not implement a final track1.txt writer until the schema is confirmed."
        )
    return {
        "found": found,
        "matches": matches,
        "candidate_schema_files": sorted(candidate_schema_files),
        "notes": notes,
        "search_terms": terms,
        "files_scanned": len(files),
    }


def write_schema_discovery_report(report: Dict[str, Any], output_path: Path) -> None:
    """Write a Markdown schema discovery report."""
    lines = []
    lines.append("# Track 1 Schema Discovery")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append("- schema_found_locally: `%s`" % bool(report.get("found")))
    lines.append("- files_scanned: `%s`" % report.get("files_scanned"))
    lines.append("- matches: `%s`" % len(report.get("matches", [])))
    lines.append("")
    lines.append("## Candidate Schema Files")
    lines.append("")
    candidates = report.get("candidate_schema_files", [])
    if candidates:
        for path in candidates:
            lines.append("- `%s`" % path)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in report.get("notes", []):
        lines.append("- %s" % note)
    lines.append("")
    lines.append("## Matches")
    lines.append("")
    matches = report.get("matches", [])
    if not matches:
        lines.append("No search-term matches were found.")
    else:
        for item in matches:
            lines.append(
                "- `%s:%s` term=`%s`: %s"
                % (
                    item.get("file_path"),
                    item.get("line_number"),
                    item.get("matched_term"),
                    _clean_markdown_line(str(item.get("line_text", ""))),
                )
            )
    lines.append("")
    lines.append("## Raw JSON")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report, indent=2, sort_keys=True))
    lines.append("```")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_text_files(root: Path, max_file_size_mb: int) -> List[Path]:
    max_size = int(max_file_size_mb) * 1024 * 1024
    files = []
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _has_skipped_dir(path):
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > max_size:
                continue
            if _is_binary(path):
                continue
        except OSError:
            continue
        files.append(path)
    return sorted(files)


def _has_skipped_dir(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def _is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:4096]
    except OSError:
        return True
    return b"\x00" in chunk


def _scan_file_for_terms(path: Path, root: Path, terms: List[str]) -> List[Dict[str, Any]]:
    matches = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return matches
    lower_terms = [(term, term.lower()) for term in terms]
    for line_number, line in enumerate(lines, start=1):
        lower = line.lower()
        for term, lower_term in lower_terms:
            if lower_term in lower:
                matches.append(
                    {
                        "file_path": _relative_path(root, path),
                        "line_number": line_number,
                        "line_text": line.strip(),
                        "matched_term": term,
                    }
                )
    return matches


def _looks_like_confirmed_schema_line(line_text: str) -> bool:
    lower = line_text.lower()
    negative_terms = [
        "todo",
        "not confirmed",
        "not implemented",
        "pending",
        "unknown",
        "do not",
        "must be confirmed",
        "schema is confirmed",
        "schema_not_confirmed",
        "official_column",
        "generic_column",
        "waiting_for_schema",
    ]
    if any(term in lower for term in negative_terms):
        return False
    schema_terms = ["schema", "columns", "delimiter", "required column", "format:", "frame indexing", "id scope"]
    official_terms = ["track1", "track 1", "submission", "official"]
    return any(term in lower for term in schema_terms) and any(term in lower for term in official_terms)


def _relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _clean_markdown_line(value: str) -> str:
    return value.replace("|", "\\|")


def _progress_iter(values: List[Any], show_progress: bool, desc: str, unit: str) -> Iterable[Any]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit=unit)


def _print_progress_iter(values: List[Any], desc: str) -> Iterable[Any]:
    total = len(values)
    for index, value in enumerate(values):
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == total:
            print("%s: %d/%d %s" % (desc, index + 1, total, value))
        yield value
