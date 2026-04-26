# GSplatTools

Dataset preparation pipeline for 3D Gaussian Splatting.
Converts raw 360° footage (DJI Osmo 360) into clean, analysis-ready image datasets.

## Project goal

**Preprocessing only — no Gaussian Splatting training code lives here.**
Training is handled by a separate tool (Lichtfeld). GSplatTools produces the clean dataset that Lichtfeld consumes.

```
eq2persp → sharp_frames → masks → [Lichtfeld for training]
```

Raw 360° footage is unsuitable for 3DGS directly. This toolset fixes that:
1. Extract perspective views from equirectangular video
2. Select only sharp, non-blurry frames
3. Generate masks to exclude dynamic objects (people, drones)

## Architecture

```
tools/
  eq2persp.py       — equirectangular → multiple perspective video clips (FFmpeg v360)
  sharp_frames.py   — blur detection + best-frame extraction (OpenCV Laplacian)
  masks.py          — dynamic object masking (SAM / YOLO)
configs/
  cameras_6view.json        — standard 6-view cube rig
  cameras_drone_orbit.json  — 8-view orbit rig, 15° down-tilt
docs/
  pipeline.md       — end-to-end workflow description
```

Each tool is a self-contained CLI script. No shared runtime state between tools.

## Typical pipeline

```bash
# Step 1: extract perspective views from 360 video
python tools/eq2persp.py footage.mp4 --views 6 --ffmpeg-path "C:/ffmpeg/bin/ffmpeg.exe"

# Step 2: extract sharpest frames from each view clip
python tools/sharp_frames.py output/footage/front/footage_front.mp4 --threshold 100

# Step 3: generate masks for dynamic objects (drone, people)
python tools/masks.py output/footage/front/frames/ --classes drone person
```

## Dependencies

Install with: `pip install -r requirements.txt`

Core:
- `opencv-python` — blur detection, image I/O
- `numpy` — array operations
- `Pillow` — image format handling
- `tqdm` — progress bars

Optional (mask generation — install when needed):
- `ultralytics` — YOLOv8 for object detection
- `torch` + `torchvision` — required by SAM
- `segment-anything` — Meta SAM for precise masks

FFmpeg (external binary, not a Python package):
- Minimum version: 4.3 (v360 filter)
- Download: https://www.gyan.dev/ffmpeg/builds/
- Configure via `--ffmpeg-path` flag or `FFMPEG_PATH` env var

## Development conventions

- Each tool is a standalone script: `if __name__ == "__main__": main()`
- stdlib only for core logic — no mandatory heavy deps at import time
- Heavy imports (torch, ultralytics) are deferred inside functions that need them
- All tools accept `--dry-run` and `-v` / `--verbose` flags
- Exit codes: 0=success, 1=partial failure, 2=fatal error
- Output always goes to `./output/<input_stem>/` unless `-o` overrides it
- Frame filenames: zero-padded 6 digits — `frame_000001.png`

## Camera: DJI Osmo 360

- Output after DJI Studio stitching: 8K H.265 equirectangular MP4, 10-bit
- Pixel format: `yuv420p10le`
- The camera itself also captures drone/operator in many shots → masking needed

## Roadmap

- [x] `eq2persp.py` — equirectangular to perspective clips
- [ ] `sharp_frames.py` — sharp frame extraction
- [ ] `masks.py` — drone/person masking
- [ ] `pipeline.py` — end-to-end orchestrator (runs all 3 steps)
- [ ] JSON-configurable pipeline runs
- [ ] Houdini / Unreal integration hooks
