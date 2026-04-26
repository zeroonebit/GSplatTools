# GSplatTools

Dataset preparation pipeline for 3D Gaussian Splatting from 360° footage.

Turns raw DJI Osmo 360 videos into clean, undistorted image datasets ready for COLMAP + 3DGS training.

## Tools

| Tool | Status | Description |
|---|---|---|
| `tools/eq2persp.py` | ✅ Ready | Extract perspective views from equirectangular video |
| `tools/sharp_frames.py` | 🚧 In progress | Select sharpest frames via blur detection |
| `tools/masks.py` | 🚧 In progress | Generate masks for drones and people |

## Requirements

- Python 3.10+
- FFmpeg ≥ 4.3 — [download](https://www.gyan.dev/ffmpeg/builds/)

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# 1. Extract 6 perspective views from a 360 clip
python tools/eq2persp.py footage.mp4 --views 6 --ffmpeg-path "C:/ffmpeg/bin/ffmpeg.exe"

# 2. Extract sharpest frames (coming soon)
python tools/sharp_frames.py output/footage/front/footage_front.mp4

# 3. Mask dynamic objects (coming soon)
python tools/masks.py output/footage/front/frames/ --classes drone person
```

## Camera

Tested with **DJI Osmo 360** — 8K equirectangular H.265, 10-bit.
Requires DJI Studio stitching before use.
