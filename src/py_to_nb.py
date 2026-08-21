"""
Convert a `# %%` cell script into a Jupyter notebook.

We author the analysis as plain .py files (readable diffs, easy to run and
debug) and generate the .ipynb deliverables from them, so the notebooks
submitted with the report are always in sync with the code that produced the
figures.

Cell markers
------------
    # %%              code cell
    # %% [markdown]   markdown cell (leading "# " is stripped from each line)

Usage
-----
    python src/py_to_nb.py notebooks/01_data_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat


def parse_cells(text: str) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    kind, buf = "code", []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            if buf:
                cells.append((kind, "\n".join(buf).strip("\n")))
                buf = []
            kind = "markdown" if "[markdown]" in stripped else "code"
        else:
            buf.append(line)
    if buf:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return [(k, s) for k, s in cells if s.strip()]


def to_notebook(py_path: Path) -> Path:
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    for kind, src in parse_cells(py_path.read_text(encoding="utf-8")):
        if kind == "markdown":
            body = "\n".join(
                line[2:] if line.startswith("# ") else line.lstrip("#")
                for line in src.splitlines()
            )
            nb.cells.append(nbformat.v4.new_markdown_cell(body.strip()))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(src))

    out_path = py_path.with_suffix(".ipynb")
    nbformat.write(nb, out_path)
    return out_path


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print("wrote", to_notebook(Path(arg)))
