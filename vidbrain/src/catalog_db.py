# src/catalog_db.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional
import struct


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS videos (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  duration REAL,
  fps REAL,
  width INTEGER,
  height INTEGER
);

CREATE TABLE IF NOT EXISTS shots (
  id INTEGER PRIMARY KEY,
  video_id INTEGER NOT NULL,
  shot_idx INTEGER NOT NULL,
  start REAL NOT NULL,
  end REAL NOT NULL,
  dur REAL NOT NULL,
  clip_path TEXT NOT NULL,

  -- NEW: optional keyframe info for debugging / preview
  frame_path TEXT,
  keyframe_sec REAL,

  UNIQUE(video_id, shot_idx),
  FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS embeddings (
  shot_id INTEGER NOT NULL,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vec BLOB NOT NULL,

  -- allow multiple models per shot
  PRIMARY KEY (shot_id, model),
  FOREIGN KEY(shot_id) REFERENCES shots(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_shots_video ON shots(video_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);
"""


def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.executescript(SCHEMA)
    return con


def upsert_video(
    con: sqlite3.Connection,
    path: str,
    duration: Optional[float] = None,
    fps: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> int:
    con.execute(
        """
        INSERT INTO videos(path, duration, fps, width, height)
        VALUES(?,?,?,?,?)
        ON CONFLICT(path) DO UPDATE SET
          duration=excluded.duration,
          fps=excluded.fps,
          width=excluded.width,
          height=excluded.height
        """,
        (path, duration, fps, width, height),
    )
    row = con.execute("SELECT id FROM videos WHERE path=?", (path,)).fetchone()
    return int(row[0])


def upsert_shot(
    con: sqlite3.Connection,
    video_id: int,
    shot_idx: int,
    start: float,
    end: float,
    clip_path: str,
    frame_path: Optional[str] = None,
    keyframe_sec: Optional[float] = None,
) -> int:
    dur = max(0.0, end - start)
    con.execute(
        """
        INSERT INTO shots(video_id, shot_idx, start, end, dur, clip_path, frame_path, keyframe_sec)
        VALUES(?,?,?,?,?,?,?,?)
        ON CONFLICT(video_id, shot_idx) DO UPDATE SET
          start=excluded.start,
          end=excluded.end,
          dur=excluded.dur,
          clip_path=excluded.clip_path,
          frame_path=excluded.frame_path,
          keyframe_sec=excluded.keyframe_sec
        """,
        (video_id, shot_idx, start, end, dur, clip_path, frame_path, keyframe_sec),
    )
    row = con.execute(
        "SELECT id FROM shots WHERE video_id=? AND shot_idx=?",
        (video_id, shot_idx),
    ).fetchone()
    return int(row[0])


def put_embedding(
    con: sqlite3.Connection,
    shot_id: int,
    model: str,
    vec: Iterable[float],
) -> None:
    v = list(vec)
    dim = len(v)
    blob = struct.pack(f"{dim}f", *v)  # float32 blob
    con.execute(
        """
        INSERT INTO embeddings(shot_id, model, dim, vec)
        VALUES(?,?,?,?)
        ON CONFLICT(shot_id, model) DO UPDATE SET
          dim=excluded.dim,
          vec=excluded.vec
        """,
        (shot_id, model, dim, blob),
    )


def get_embedding(con: sqlite3.Connection, shot_id: int, model: str) -> Optional[list[float]]:
    row = con.execute(
        "SELECT dim, vec FROM embeddings WHERE shot_id=? AND model=?",
        (shot_id, model),
    ).fetchone()
    if not row:
        return None
    dim, blob = int(row[0]), row[1]
    return list(struct.unpack(f"{dim}f", blob))