"""Reporting package: turn a :class:`~forensics.inference.run_types.RunResult` into
accurate, beautiful, *traceable* output.

Public entry point (the CLI depends on this exact signature)::

    build_report(run, out_dir, formats=('md','json','html')) -> dict[str, str]

It writes the requested formats into ``out_dir`` and returns ``{format: filepath}``:

* ``md``   — full Markdown report (sections A-L) with embedded Mermaid diagrams,
* ``json`` — the whole RunResult via :func:`forensics.core.types.to_jsonable`,
* ``html`` — a self-contained interactive single-file report where every diagram node
  and table row is clickable to reveal how it was inferred.

The heavy lifting lives in :mod:`.diagrams`, :mod:`.charts`, :mod:`.markdown`,
:mod:`.html` and :mod:`.style`; this module just orchestrates and writes files.
"""
from __future__ import annotations

import json
import os
from typing import Iterable

from forensics.core.types import to_jsonable
from forensics.inference.run_types import RunResult

from . import charts as _charts
from . import diagrams as _diagrams
from . import html as _html
from . import markdown as _markdown

__all__ = ["build_report", "write_report"]

_FILENAMES = {"md": "report.md", "json": "report.json", "html": "report.html"}


def build_report(run: RunResult, out_dir: str,
                 formats: Iterable[str] = ("md", "json", "html")) -> dict[str, str]:
    """Render ``run`` into ``out_dir`` in each requested format; return {format: path}.

    Diagrams and charts are built once and shared across the Markdown and HTML outputs.
    Unknown format names are ignored. Missing/None fields in the run (e.g. an
    unrecovered ``hidden_size``) degrade gracefully to explicit "not identified" output.
    """
    formats = list(formats)
    os.makedirs(out_dir, exist_ok=True)
    written: dict[str, str] = {}

    diags = None
    charts = None
    need_visuals = any(f in ("md", "html") for f in formats)
    if need_visuals:
        diags = _diagrams.all_diagrams(run)
    if "html" in formats:
        charts = _charts.all_charts(run)

    for fmt in formats:
        fname = _FILENAMES.get(fmt)
        if not fname:
            continue
        path = os.path.join(out_dir, fname)
        if fmt == "md":
            text = _markdown.build_markdown(run, diags=diags)
            _write_text(path, text)
        elif fmt == "json":
            payload = to_jsonable(run)
            _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2,
                                         default=_html._json_default))
        elif fmt == "html":
            text = _html.build_html(run, diags=diags, charts=charts)
            _write_text(path, text)
        else:
            continue
        written[fmt] = path

    return written


# convenience alias some callers may prefer
write_report = build_report


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
