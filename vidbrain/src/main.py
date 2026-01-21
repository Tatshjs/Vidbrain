# src/main.py
import argparse
import json
from pathlib import Path

from shot_detect import detect_shots
from cut_shots import cut_shots
from embed_shots import embed_shots
from catalog_build import build_catalog


def main():
    p = argparse.ArgumentParser(prog="vidbrain")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- shots ----
    shots = sub.add_parser("shots", help="Detect shots (scene cuts) and write JSON index.")
    shots.add_argument("video", help="Path to input video file")
    shots.add_argument("--out", default="output/shots", help="Output folder for shot index json")
    shots.add_argument("--min-shot", type=float, default=1.0, help="Minimum shot duration (sec); shorter gets merged")
    shots.add_argument("--max-shot", type=float, default=8.0, help="Maximum shot duration (sec); longer gets split")
    shots.add_argument("--threshold", type=float, default=27.0, help="Cut threshold (higher = fewer cuts)")
    shots.add_argument("--fps-sample", type=int, default=2, help="Process every Nth frame (speed vs accuracy)")

    # ---- cut ----
    cut = sub.add_parser("cut", help="Cut detected shots into individual mp4 clips using ffmpeg.")
    cut.add_argument("shots_json", help="Path to *.shots.json (output of 'shots')")
    cut.add_argument("--out", default="output/clips", help="Output folder for clips")
    cut.add_argument("--copy", action="store_true", help="Use stream copy (faster, less reliable). Default: re-encode")

    # ---- embed ----
    emb = sub.add_parser("embed", help="Compute CLIP embeddings for each clip and write JSON.")
    emb.add_argument("clips_dir", help="Path to folder with shot clips (output of 'cut')")
    emb.add_argument(
        "--out",
        default="output",
        help="Output base folder (will write output/frames and output/embeddings)",
    )
    emb.add_argument("--model", default="ViT-L-14", help="OpenCLIP model name")
    emb.add_argument("--pretrained", default="openai", help="OpenCLIP pretrained tag")
    emb.add_argument("--device", default=None, help="cpu|cuda (default: auto)")

    # ---- catalog ----
    cat = sub.add_parser("catalog", help="Build output/catalog.db from embeddings json.")
    cat.add_argument("--emb", default="output/embeddings/shot_embeddings.json", help="Embeddings json")
    cat.add_argument("--db", default="output/catalog.db", help="SQLite db output")
    cat.add_argument("--model", default="clip", help="Model label stored in DB")

    # ---- index (one command) ----
    idx = sub.add_parser("index", help="Run shots -> cut -> embed -> catalog for one video.")
    idx.add_argument("video", help="Path to input video file")
    idx.add_argument("--shots-out", default="output/shots", help="Output folder for shot index json")
    idx.add_argument("--clips-out", default="output/clips", help="Output folder for clips")
    idx.add_argument(
        "--out",
        default="output",
        help="Output base folder (frames + embeddings + catalog.db live here)",
    )
    idx.add_argument("--db", default="output/catalog.db", help="SQLite db output")

    idx.add_argument("--min-shot", type=float, default=1.0, help="Minimum shot duration (sec)")
    idx.add_argument("--max-shot", type=float, default=8.0, help="Maximum shot duration (sec)")
    idx.add_argument("--threshold", type=float, default=27.0, help="Cut threshold")
    idx.add_argument("--fps-sample", type=int, default=2, help="Process every Nth frame")
    idx.add_argument("--copy", action="store_true", help="Use stream copy for cutting (default re-encode)")

    idx.add_argument("--clip-model", default="ViT-L-14", help="OpenCLIP model name")
    idx.add_argument("--pretrained", default="openai", help="OpenCLIP pretrained tag")
    idx.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    idx.add_argument("--model-label", default="clip", help="Label stored in DB for embeddings")

    args = p.parse_args()

    if args.cmd == "shots":
        video_path = Path(args.video)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        index = detect_shots(
            video_path=video_path,
            min_shot_sec=args.min_shot,
            max_shot_sec=args.max_shot,
            threshold=args.threshold,
            frame_skip=args.fps_sample,
        )

        out_file = out_dir / f"{video_path.stem}.shots.json"
        out_file.write_text(json.dumps(index, indent=2), encoding="utf-8")
        print(f"Wrote: {out_file}")
        print(f"Shots: {len(index['shots'])}")
        return

    if args.cmd == "cut":
        shots_json = Path(args.shots_json)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        clip_dir = cut_shots(
            shots_json=shots_json,
            out_dir=out_dir,
            reencode=not args.copy,
        )
        print(f"OK: wrote clips to {clip_dir}")
        return

    if args.cmd == "embed":
        clips_dir = Path(args.clips_dir)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = embed_shots(
            clips_dir=clips_dir,
            out_dir=out_dir,
            model_name=args.model,
            pretrained=args.pretrained,
            device=args.device,
        )
        print(f"Wrote: {out_file}")
        return

    if args.cmd == "catalog":
        build_catalog(Path(args.emb), Path(args.db), args.model)
        print(f"Wrote: {args.db}")
        return

    if args.cmd == "index":
        video_path = Path(args.video)

        # 1) shots
        shots_out = Path(args.shots_out)
        shots_out.mkdir(parents=True, exist_ok=True)
        index = detect_shots(
            video_path=video_path,
            min_shot_sec=args.min_shot,
            max_shot_sec=args.max_shot,
            threshold=args.threshold,
            frame_skip=args.fps_sample,
        )
        shots_json = shots_out / f"{video_path.stem}.shots.json"
        shots_json.write_text(json.dumps(index, indent=2), encoding="utf-8")
        print(f"Wrote: {shots_json} (shots={len(index['shots'])})")

        # 2) cut
        clips_out = Path(args.clips_out)
        clips_out.mkdir(parents=True, exist_ok=True)
        clip_dir = cut_shots(shots_json=shots_json, out_dir=clips_out, reencode=not args.copy)
        print(f"OK: wrote clips to {clip_dir}")

        # 3) embed
        out_base = Path(args.out)
        out_base.mkdir(parents=True, exist_ok=True)
        emb_json = embed_shots(
            clips_dir=Path(clip_dir),
            out_dir=out_base,
            model_name=args.clip_model,
            pretrained=args.pretrained,
            device=args.device,
        )
        print(f"Wrote: {emb_json}")

        # 4) catalog
        build_catalog(Path(emb_json), Path(args.db), args.model_label)
        print(f"Wrote DB: {args.db}")
        return


if __name__ == "__main__":
    main()