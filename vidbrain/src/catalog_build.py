from pathlib import Path
import json
import sqlite3
from typing import Dict, Any


def _extract_items(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts:
    - { "items": { clip_id: {...}, ... } }
    - { "shots": [ {...}, {...} ] }
    - { clip_id: {...}, ... }  (flat dict)
    Returns: dict clip_id -> item
    """
    if "items" in data and isinstance(data["items"], dict):
        return data["items"]

    if "shots" in data and isinstance(data["shots"], list):
        return {str(i): s for i, s in enumerate(data["shots"])}

    # flat dict fallback (clip -> dict)
    if all(isinstance(v, dict) for v in data.values()):
        return data

    raise ValueError(
        "Could not find 'items' or 'shots' list (or mapping of clip->dict) in embeddings json."
    )


def build_catalog(embeddings_json: Path, db_path: Path, model_label: str = "clip"):
    embeddings_json = Path(embeddings_json)
    db_path = Path(db_path)

    data = json.loads(embeddings_json.read_text(encoding="utf-8"))
    items = _extract_items(data)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS clips (
            id TEXT PRIMARY KEY,
            path TEXT,
            frame TEXT,
            keyframe_sec REAL,
            model TEXT,
            embedding BLOB
        )
    """)

    for clip_id, item in items.items():
        c.execute(
            "INSERT OR REPLACE INTO clips VALUES (?, ?, ?, ?, ?, ?)",
            (
                clip_id,
                item.get("clip"),
                item.get("frame"),
                item.get("keyframe_sec", 0.0),
                model_label,
                json.dumps(item.get("embedding")),
            ),
        )

    conn.commit()
    conn.close()
    print(f"Catalog DB written: {db_path}")