"""The model's lab notebook: write_down appends, read_notebook reads it all.

Each entry is kept in memory and appended to two files as it is written,
so the notebook survives a crash: notebook.md for reading, notebook.jsonl
for analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_ENTRY_CHARS = 8000


@dataclass
class NotebookEntry:
    number: int
    call: int          # model call on which it was written
    elapsed_s: float
    text: str


class Notebook:
    def __init__(self, folder: Path | None = None):
        self.entries: list[NotebookEntry] = []
        self._started = time.monotonic()
        self._md = folder / "notebook.md" if folder else None
        self._jsonl = folder / "notebook.jsonl" if folder else None

    def write(self, text: str, call: int) -> NotebookEntry:
        text = str(text).strip()
        if len(text) > MAX_ENTRY_CHARS:
            text = text[:MAX_ENTRY_CHARS] + " [...cut: entry too long]"
        entry = NotebookEntry(
            number=len(self.entries) + 1,
            call=call,
            elapsed_s=round(time.monotonic() - self._started, 1),
            text=text,
        )
        self.entries.append(entry)
        if self._md is not None:
            with self._md.open("a", encoding="utf-8") as f:
                f.write(f"## Entry {entry.number} (call {entry.call})\n\n{entry.text}\n\n")
        if self._jsonl is not None:
            with self._jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        return entry

    def read(self) -> str:
        if not self.entries:
            return "The notebook is empty."
        return "\n\n".join(f"Entry {e.number}:\n{e.text}" for e in self.entries)
