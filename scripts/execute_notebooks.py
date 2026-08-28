"""Execute notebooks as an integration test without modifying their sources."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    arguments = parser.parse_args()
    paths = arguments.paths or sorted(Path("notebooks").glob("*.ipynb"))
    failures = []
    for path in paths:
        print(f"Executing {path}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        try:
            NotebookClient(
                notebook,
                timeout=arguments.timeout,
                kernel_name="python3",
                resources={"metadata": {"path": str(path.parent.resolve())}},
            ).execute()
        except Exception as exc:  # pragma: no cover - integration helper
            failures.append((path, exc))
            print(f"FAILED {path}: {exc}", flush=True)
        else:
            print(f"PASSED {path}", flush=True)
    if failures:
        print("Notebook failures:")
        for path, error in failures:
            print(f"- {path}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

