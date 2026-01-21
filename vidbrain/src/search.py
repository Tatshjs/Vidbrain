from __future__ import annotations

from pathlib import Path
import sqlite3
import json
import math
from typing import List, Tuple, Optional

import torch
import open_clip


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-8)


def _resolve_device(device: str) -> str:
    device = (device or "cpu").lower().strip()
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return device


def _load_text_model(model_name: str, pretrained: str, device: str):
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer


def _embed_text(model, tokenizer, text: str, device: str) -> List[float]:
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].detach().cpu().tolist()


def _db_get_model_info(conn: sqlite3.Connection) -> Tuple[Optional[str], Optional[str]]:
    """
    Reads model info from DB if present.
    New schema: clips(model_name, pretrained)
    Old schema: clips(model)
    """
    c = conn.cursor()
    cols = [row[1] for row in c.execute("PRAGMA table_info(clips)").fetchall()]
    cols_set = set(cols)

    if "model_name" in cols_set and "pretrained" in cols_set:
        row = c.execute(
            "SELECT model_name, pretrained FROM clips WHERE model_name IS NOT NULL LIMIT 1"
        ).fetchone()
        if row and isinstance(row[0], str):
            return row[0], row[1] if isinstance(row[1], str) else None

    if "model" in cols_set:
        row = c.execute("SELECT model FROM clips WHERE model IS NOT NULL LIMIT 1").fetchone()
        if row and isinstance(row[0], str):
            return row[0], None

    return None, None


def search_catalog(
    db_path: Path,
    query: str,
    model_name: Optional[str] = None,
    pretrained: Optional[str] = None,
    device: str = "cpu",
    top_k: int = 10,
):
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Catalog DB not found: {db_path}")

    device = _resolve_device(device)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Auto-pick model + pretrained from DB if not provided
    db_model, db_pretrained = _db_get_model_info(conn)
    if not model_name:
        model_name = db_model or "ViT-L-14"
    if not pretrained:
        pretrained = db_pretrained or "openai"

    # Load text encoder with same CLIP family
    model, tokenizer = _load_text_model(model_name, pretrained, device)
    q_emb = _embed_text(model, tokenizer, query, device)

    # Select correct columns depending on schema
    cols = {row[1] for row in c.execute("PRAGMA table_info(clips)").fetchall()}
    if "embedding" not in cols:
        conn.close()
        raise RuntimeError("DB schema invalid: 'embedding' column missing in clips table.")

    if "frame" in cols:
        rows = c.execute("SELECT id, path, frame, embedding FROM clips").fetchall()
        has_frame = True
    else:
        rows = c.execute("SELECT id, path, embedding FROM clips").fetchall()
        has_frame = False

    conn.close()

    results: List[Tuple[float, str, str, str]] = []

    if has_frame:
        for clip_id, path, frame, emb_json in rows:
            emb = json.loads(emb_json)
            score = _cosine(q_emb, emb)
            results.append((score, str(clip_id), str(path), str(frame)))
    else:
        for clip_id, path, emb_json in rows:
            emb = json.loads(emb_json)
            score = _cosine(q_emb, emb)
            results.append((score, str(clip_id), str(path), ""))

    results.sort(reverse=True, key=lambda x: x[0])
    return results[:top_k]


def print_results(results):
    for i, (score, clip_id, path, frame) in enumerate(results, 1):
        print(f"{i:02d}. {score:.3f}  {clip_id}")
        print(f"     clip : {path}")
        if frame:
            print(f"     frame: {frame}")
        print()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(prog="vidbrain-search")
    p.add_argument("query", help="Search text (e.g. 'dark bunker people fear')")
    p.add_argument("--db", default="output/catalog.db", help="Path to catalog.db")
    p.add_argument("--top", type=int, default=10, help="How many results to show")
    p.add_argument("--model", default=None, help="Override CLIP model name (else auto from DB)")
    p.add_argument("--pretrained", default=None, help="Override pretrained tag (else auto from DB)")
    p.add_argument("--device", default="cpu", help="cpu|cuda (default: cpu)")
    args = p.parse_args()

    res = search_catalog(
        db_path=Path(args.db),
        query=args.query,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        top_k=args.top,
    )
    print_results(res)