from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class Shot:
    id: int
    start: float
    end: float
    dur: float


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDERR:\n"
            + (p.stderr[-4000:] if p.stderr else "")
        )


def load_shots_json(path: Path) -> tuple[Path, List[Shot]]:
    data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    video = Path(data["video"])
    shots = []
    for s in data["shots"]:
        shots.append(
            Shot(
                id=int(s["id"]),
                start=float(s["start"]),
                end=float(s["end"]),
                dur=float(s["dur"]),
            )
        )
    return video, shots


def cut_shots(
    shots_json: Path,
    out_dir: Path,
    reencode: bool = True,
    vcodec: str = "libx264",
    crf: int = 20,
    preset: str = "veryfast",
    acodec: str = "aac",
    abitrate: str = "192k",
) -> Path:
    video_path, shots = load_shots_json(shots_json)

    if not shots_json.exists():
        raise FileNotFoundError(str(shots_json))
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Name folder after video stem to avoid collisions
    clip_dir = out_dir / video_path.stem
    clip_dir.mkdir(parents=True, exist_ok=True)

    for sh in shots:
        out_file = clip_dir / f"shot_{sh.id:03d}.mp4"

        # Re-encode is safest (works even if keyframes are weird).
        if reencode:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-v",
                "error",
                "-ss",
                f"{sh.start:.3f}",
                "-t",
                f"{sh.dur:.3f}",
                "-i",
                str(video_path),
                "-c:v",
                vcodec,
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-c:a",
                acodec,
                "-b:a",
                abitrate,
                str(out_file),
            ]
        else:
            # Faster, but can fail or be inaccurate if not on keyframes.
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-v",
                "error",
                "-ss",
                f"{sh.start:.3f}",
                "-t",
                f"{sh.dur:.3f}",
                "-i",
                str(video_path),
                "-c",
                "copy",
                str(out_file),
            ]

        _run(cmd)

    # Write a small index file for later steps
    index = {
        "video": str(video_path),
        "shots_json": str(shots_json),
        "clip_dir": str(clip_dir),
        "count": len(shots),
    }
    index_path = clip_dir / "clips.index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    return clip_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("shots_json", type=str, help="Path to *.shots.json")
    ap.add_argument("--out", type=str, default="output/clips", help="Output base folder")
    ap.add_argument("--copy", action="store_true", help="Try stream copy (faster, less reliable)")
    args = ap.parse_args()

    clip_dir = cut_shots(
        Path(args.shots_json),
        Path(args.out),
        reencode=not args.copy,
    )
    print(f"OK: wrote clips to {clip_dir}")