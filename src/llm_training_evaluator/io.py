from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import Sample


def load_samples(path: str | Path) -> list[Sample]:
    source = Path(path)
    samples: list[Sample] = []
    seen_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if "prompt" not in record:
                raise ValueError(f"Missing 'prompt' at {source}:{line_number}")
            sample_id = str(record.get("id", f"sample_{line_number:06d}"))
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample id {sample_id!r} at {source}:{line_number}")
            seen_ids.add(sample_id)
            samples.append(
                Sample(
                    id=sample_id,
                    prompt=str(record["prompt"]),
                    target=(str(record["target"]) if record.get("target") is not None else None),
                    tags=[str(tag) for tag in record.get("tags", [])],
                    metadata=dict(record.get("metadata", {})),
                )
            )
    if not samples:
        raise ValueError(f"No samples found in {source}")
    return samples


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(destination)
    return destination


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
