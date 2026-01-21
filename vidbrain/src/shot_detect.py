from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


@dataclass
class Shot:
    start: float
    end: float

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


def _to_seconds(timecode) -> float:
    # scenedetect timecode has get_seconds()
    return float(timecode.get_seconds())


def _merge_short(shots: List[Shot], min_sec: float) -> List[Shot]:
    if not shots:
        return shots
    merged: List[Shot] = [shots[0]]
    for s in shots[1:]:
        last = merged[-1]
        if last.dur < min_sec:
            # merge last into current (extend last)
            last.end = s.end
        elif s.dur < min_sec:
            # merge current into last
            last.end = s.end
        else:
            merged.append(s)
    return merged


def _split_long(shots: List[Shot], max_sec: float) -> List[Shot]:
    out: List[Shot] = []
    for s in shots:
        if s.dur <= max_sec:
            out.append(s)
            continue
        # split into equal-ish chunks up to max_sec
        t = s.start
        while t < s.end:
            end = min(s.end, t + max_sec)
            out.append(Shot(start=t, end=end))
            t = end
    return out


def detect_shots(
    video_path: Path,
    min_shot_sec: float = 1.0,
    max_shot_sec: float = 8.0,
    threshold: float = 27.0,
    frame_skip: int = 2,
) -> Dict:
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    video_manager = VideoManager([str(video_path)])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # frame_skip=2 => process every 2nd frame (faster)
    video_manager.set_downscale_factor()  # auto downscale for speed
    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager, frame_skip=frame_skip)
    scene_list = scene_manager.get_scene_list()

    shots: List[Shot] = []
    if scene_list:
        for start_tc, end_tc in scene_list:
            shots.append(Shot(start=_to_seconds(start_tc), end=_to_seconds(end_tc)))
    else:
        # no cuts found => one shot whole video
        duration = video_manager.get_duration().get_seconds()
        shots.append(Shot(start=0.0, end=float(duration)))

    video_manager.release()

    # post rules: merge very short, split very long
    shots = _merge_short(shots, min_shot_sec)
    shots = _split_long(shots, max_shot_sec)

    # final cleanup: remove zero/negative
    shots = [s for s in shots if s.dur > 0.05]

    return {
        "video": str(video_path),
        "threshold": threshold,
        "min_shot_sec": min_shot_sec,
        "max_shot_sec": max_shot_sec,
        "frame_skip": frame_skip,
        "shots": [
            {
                "id": i,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "dur": round(s.dur, 3),
            }
            for i, s in enumerate(shots)
        ],
    }