import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from core.entities import InteractionEvent


def save_interaction_events(
    events: Iterable[InteractionEvent],
    output_path: str | Path,
    metadata: dict | None = None,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": "interaction_events.v1",
        "metadata": metadata or {},
        "events": [asdict(event) for event in events],
    }

    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    return str(path)


def load_interaction_events(input_path: str | Path) -> List[InteractionEvent]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if payload.get("schema") != "interaction_events.v1":
        raise ValueError(f"Unsupported event dataset schema in {path}")

    return [InteractionEvent(**item) for item in payload.get("events", [])]
