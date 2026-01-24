from __future__ import annotations

from pathlib import Path
import json
import sqlite3
from typing import Dict, Any, Tuple


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
    if isinstance(data, dict) and data and all(isinstance(v, dict) for v in data.values()):
        return data

    raise ValueError(
        "Could not find 'items' or 'shots' list (or mapping of clip->dict) in embeddings json."
    )


def _extract_model_meta(data: Dict[str, Any], fallback_label: str) -> Tuple[str, str]:
    """
    Tries to read model info from embeddings json:
    { "model": { "name": "...", "pretrained": "..." } }
    Returns: (model_name, pretrained)
    """
    m = data.get("model")
    if isinstance(m, dict):
        name = m.get("name")
        pretrained = m.get("pretrained")
        if isinstance(name, str) and isinstance(pretrained, str) and name and pretrained:
            return name, pretrained

    # fallback: keep something useful in DB
    return fallback_label, "unknown"


def build_catalog(embeddings_json: Path, db_path: Path, model_label: str = "clip") -> None:
    embeddings_json = Path(embeddings_json)
    db_path = Path(db_path)

    data = json.loads(embeddings_json.read_text(encoding="utf-8"))
    items = _extract_items(data)
    model_name, pretrained = _extract_model_meta(data, model_label)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # New schema: store model_name + pretrained so search can auto-match correctly.
    c.execute("""
        CREATE TABLE IF NOT EXISTS clips (
            id TEXT PRIMARY KEY,
            path TEXT,
            frame TEXT,
            keyframe_sec REAL,
            model_name TEXT,
            pretrained TEXT,
            embedding TEXT
        )
    """)

    # Backward-compat: if an old DB exists with column "model" but not "model_name",
    # try to migrate quickly.
    cols = {row[1] for row in c.execute("PRAGMA table_info(clips)").fetchall()}
    if "model" in cols and "model_name" not in cols:
        # old layout detected -> rebuild table
        c.execute("ALTER TABLE clips RENAME TO clips_old")
        c.execute("""
            CREATE TABLE clips (
                id TEXT PRIMARY KEY,
                path TEXT,
                frame TEXT,
                keyframe_sec REAL,
                model_name TEXT,
                pretrained TEXT,
                embedding TEXT
            )
        """)
        # best-effort migrate old rows
        old_rows = c.execute("SELECT id, path, frame, keyframe_sec, model, embedding FROM clips_old").fetchall()
        for clip_id, path, frame, keyframe_sec, old_model, embedding in old_rows:
            c.execute(
                "INSERT OR REPLACE INTO clips VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    clip_id,
                    path,
                    frame,
                    float(keyframe_sec or 0.0),
                    str(old_model or model_name),
                    "unknown",
                    embedding,
                ),
            )
        c.execute("DROP TABLE clips_old")

    for clip_id, item in items.items():
        c.execute(
            "INSERT OR REPLACE INTO clips VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                clip_id,
                item.get("clip"),
                item.get("frame"),
                float(item.get("keyframe_sec", 0.0) or 0.0),
                model_name,
                pretrained,
                json.dumps(item.get("embedding")),
            ),
        )

    conn.commit()
    conn.close()
    print(f"Catalog DB written: {db_path}")