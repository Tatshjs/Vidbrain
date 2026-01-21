# src/query.py
from __future__ import annotations

import argparse
import sqlite3
import struct
import subprocess
from pathlib import Path
from typing import Iterable, Tuple, Optional, List

import numpy as np
import torch
import open_clip


# -----------------------
# CLIP text embedding
# -----------------------
def load_text_model(model_name: str, pretrained: str, device: str):
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    return model


def embed_text(model, text: str, device: str) -> np.ndarray:
    with torch.no_grad():
        tokens = open_clip.tokenize([text]).to(device)
        emb = model.encode_text(tokens)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
    v = emb[0].detach().cpu().to(torch.float32).numpy()
    return v


# -----------------------
# DB helpers (BLOB -> vec)
# -----------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def unpack_f32_blob(blob: bytes, dim: int) -> np.ndarray:
    # blob is packed float32: struct.pack(f"{dim}f", *vec)
    fmt = f"{dim}f"
    vec = struct.unpack(fmt, blob)
    return np.asarray(vec, dtype=np.float32)


def db_has_table(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def detect_db_schema(con: sqlite3.Connection) -> str:
    """
    Return "new" for videos/shots/embeddings schema,
    return "old" for legacy clips table,
    else raise.
    """
    if db_has_table(con, "shots") and db_has_table(con, "embeddings"):
        return "new"
    if db_has_table(con, "clips"):
        return "old"
    raise RuntimeError("DB schema not recognized (expected tables shots+embeddings or clips).")


def fetch_dims_and_model(con: sqlite3.Connection) -> Tuple[int, str]:
    """
    From new schema: embeddings.dim + embeddings.model exist.
    We take the first row.
    """
    row = con.execute("SELECT dim, model FROM embeddings LIMIT 1").fetchone()
    if not row:
        raise RuntimeError("No embeddings found in DB.")
    dim = int(row[0])
    model_label = str(row[1])
    return dim, model_label


def infer_openclip_from_dim(dim: int) -> Tuple[str, str]:
    # keep it simple / safe defaults
    if dim == 512:
        return ("ViT-B-32", "laion2b_s34b_b79k")
    if dim == 768:
        return ("ViT-L-14", "openai")
    # fallback
    return ("ViT-B-32", "laion2b_s34b_b79k")


def iter_items_new_schema(
    con: sqlite3.Connection, model_filter: Optional[str] = None
) -> Iterable[Tuple[str, int, float, float, int, bytes]]:
    """
    Yields:
      (clip_path, shot_idx, start, end, dim, vec_blob)
    """
    if model_filter:
        q = """
        SELECT s.clip_path, s.shot_idx, s.start, s.end, e.dim, e.vec
        FROM shots s
        JOIN embeddings e ON e.shot_id = s.id
        WHERE e.model = ?
        ORDER BY s.id
        """
        cur = con.execute(q, (model_filter,))
    else:
        q = """
        SELECT s.clip_path, s.shot_idx, s.start, s.end, e.dim, e.vec
        FROM shots s
        JOIN embeddings e ON e.shot_id = s.id
        ORDER BY s.id
        """
        cur = con.execute(q)

    for row in cur:
        yield (row[0], int(row[1]), float(row[2]), float(row[3]), int(row[4]), row[5])


# -----------------------
# main
# -----------------------
def main():
    p = argparse.ArgumentParser(prog="vidbrain-query")
    p.add_argument("query", help='Text query, e.g. "face close up"')
    p.add_argument("--db", default="output/catalog.db", help="Path to catalog SQLite DB")
    p.add_argument("--top", type=int, default=10, help="Number of results")
    p.add_argument("--open", action="store_true", help="Open best matches via xdg-open")
    p.add_argument("--device", default="cpu", help="cpu (codespaces usually no cuda)")
    p.add_argument("--model", default=None, help='Override CLIP model name (e.g. "ViT-L-14")')
    p.add_argument("--pretrained", default=None, help='Override pretrained tag (e.g. "openai")')
    p.add_argument("--db-model", default=None, help="Optional filter for embeddings.model inside DB (e.g. clip)")
    args = p.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path.resolve()}")

    con = sqlite3.connect(str(db_path))
    try:
        schema = detect_db_schema(con)

        if schema != "new":
            raise RuntimeError(
                "Your DB is not using the new schema (shots/embeddings).\n"
                "Rebuild your DB with the current toolchain (main.py index) so query works."
            )

        db_dim, db_model_label = fetch_dims_and_model(con)

        # Pick CLIP model that matches dimension unless overridden
        if args.model and args.pretrained:
            model_name, pretrained = args.model, args.pretrained
        else:
            model_name, pretrained = infer_openclip_from_dim(db_dim)

        # Text model
        model = load_text_model(model_name, pretrained, args.device)
        q = embed_text(model, args.query, args.device)

        if q.shape[0] != db_dim:
            raise RuntimeError(
                f"Model mismatch: text dim={q.shape[0]} but DB dim={db_dim}.\n"
                f"Fix: run query with matching model OR rebuild embeddings with same model.\n"
                f"Suggestion for dim={db_dim}: --model {model_name} --pretrained {pretrained}"
            )

        # Stream items and keep only top-k (so RAM stays tiny)
        top_k = args.top
        best: List[Tuple[float, str, int, float, float]] = []

        model_filter = args.db_model or None
        for clip_path, shot_idx, start, end, dim, blob in iter_items_new_schema(con, model_filter=model_filter):
            if dim != db_dim:
                continue
            emb = unpack_f32_blob(blob, dim)
            s = cosine(q, emb)

            if len(best) < top_k:
                best.append((s, clip_path, shot_idx, start, end))
                best.sort(key=lambda x: x[0], reverse=True)
            else:
                if s > best[-1][0]:
                    best[-1] = (s, clip_path, shot_idx, start, end)
                    best.sort(key=lambda x: x[0], reverse=True)

        print(f'\nQuery: "{args.query}"')
        print(f"DB: {db_path} | dim={db_dim} | db_model={db_model_label}")
        print(f"Text model: {model_name} ({pretrained})\n")

        for i, (s, clip, shot_idx, start, end) in enumerate(best, start=1):
            name = Path(clip).name
            print(f"{i:02d} | {s:.3f} | {name} | shot={shot_idx} | {start:.2f}-{end:.2f}s")
            if args.open:
                subprocess.Popen(["xdg-open", clip])

    finally:
        con.close()


if __name__ == "__main__":
    main()