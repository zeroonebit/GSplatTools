Review the GSplatTools output for a scene.

Arguments: $ARGUMENTS (scene stem or full path, e.g. 0426 or output/0426)

Check and report:
1. Which views exist under output/<stem>/
2. Frame count per view (count files in frames/ subdirs)
3. Mask coverage per view (count files in masks/ subdirs vs frames)
4. Whether colmap/ exists and what models were produced
5. Any missing steps (e.g. has clips but no frames = sharp_frames not run)
6. Recommended next step

Keep the report concise — one line per item.
