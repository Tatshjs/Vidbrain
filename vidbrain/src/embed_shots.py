from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
from typing import Dict, Any, Optional

from PIL import Image
import torch
import open_clip
from tqdm import tqdm


@dataclass
class ClipItem:
    stem: str
    clip_path: Path
    frame_path: Path
    embedding: list[float]


def _run(cmd: list[str]) -> None:
    """Run a command and raise a readable error if it fails."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(cmd)}\n\n"
            f"STDERR:\n{p.stderr}\n"
        )


def _ffprobe_duration(video_path: Path) -> float:
    """Return duration in seconds (float)."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        return 2.0
    try:
        return float(p.stdout.strip())
    except Exception:
        return 2.0


def extract_keyframe(video_path: Path, out_path: Path, at_sec: Optional[float] = None) -> float:
    """
    Extract a single keyframe JPG.
    If at_sec is None -> use middle of clip.
    Returns timestamp used.
    """
    dur = _ffprobe_duration(video_path)
    if at_sec is None:
        at_sec = max(0.0, dur / 2.0)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{at_sec:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(out_path),
    ]
    _run(cmd)
    return at_sec


def _resolve_device(requested: Optional[str]) -> str:
    """
    Make device selection robust.
    In Codespaces torch may be installed with CUDA packages but no NVIDIA driver exists.
    """
    if requested is None:
        # auto
        return "cuda" if torch.cuda.is_available() else "cpu"

    req = str(requested).strip().lower()
    if req in ("cuda", "gpu"):
        return "cuda"
    return "cpu"


def load_clip(model_name: str, pretrained: str, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )

    # Try requested device, but fall back to CPU if CUDA is not actually usable.
    try:
        model = model.to(device)
    except RuntimeError as e:
        # Typical: "Found no NVIDIA driver on your system"
        print(f"[WARN] Device '{device}' not usable ({e}). Falling back to CPU.")
        device = "cpu"
        model = model.to(device)

    model.eval()
    return model, preprocess, device


def embed_image(model, preprocess, image_path: Path, device: str) -> list[float]:
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].detach().cpu().tolist()


def embed_shots(
    clips_dir: Path,
    out_dir: Path,
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    device: Optional[str] = None,
) -> Path:
    """
    Build a vision index for all *.mp4 in clips_dir.
    Writes:
      out_dir/frames/<clip>.jpg
      out_dir/embeddings/shot_embeddings.json
    Returns path to shot_embeddings.json
    """
    clips_dir = Path(clips_dir)
    out_dir = Path(out_dir)

    device = _resolve_device(device)

    frames_dir = out_dir / "frames"
    embed_dir = out_dir / "embeddings"
    frames_dir.mkdir(parents=True, exist_ok=True)
    embed_dir.mkdir(parents=True, exist_ok=True)

    model, preprocess, device = load_clip(model_name, pretrained, device)

    index: Dict[str, Any] = {
        "model": {"name": model_name, "pretrained": pretrained, "device": device},
        "clips_dir": str(clips_dir),
        "items": {},
    }

    clips = sorted(clips_dir.glob("*.mp4"))
    if not clips:
        raise FileNotFoundError(f"No .mp4 clips found in: {clips_dir}")

    for clip in tqdm(clips, desc="Embedding shots"):
        frame_path = frames_dir / f"{clip.stem}.jpg"

        # Extract keyframe (middle of clip)
        t = extract_keyframe(clip, frame_path, at_sec=None)

        # Embed
        embedding = embed_image(model, preprocess, frame_path, device)

        index["items"][clip.stem] = {
            "clip": str(clip),
            "frame": str(frame_path),
            "keyframe_sec": t,
            "embedding": embedding,
        }

    out_file = embed_dir / "shot_embeddings.json"
    out_file.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Vision index written to: {out_file}")
    return out_file


def process_clips(clips_dir: Path, out_dir: Path):
    # Backwards compatible alias
    return embed_shots(clips_dir=clips_dir, out_dir=out_dir)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(prog="vidbrain-embed")
    p.add_argument("clips", help="Folder with shot clips (mp4)")
    p.add_argument("--out", default="output", help="Output base folder")
    p.add_argument("--model", default="ViT-L-14", help="OpenCLIP model name")
    p.add_argument("--pretrained", default="openai", help="OpenCLIP pretrained tag")
    p.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    args = p.parse_args()

    embed_shots(
        clips_dir=Path(args.clips),
        out_dir=Path(args.out),
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
    )