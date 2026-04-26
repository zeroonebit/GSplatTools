Process a 360° video through the full GSplatTools pipeline.

Arguments: $ARGUMENTS (video file path, e.g. H:\footage\clip.mp4)

Steps:
1. Verify the video file exists and is readable
2. Run eq2persp dry-run to show what views will be created
3. Check if output/<stem>/ already has partial results and report
4. Suggest the full pipeline command with appropriate flags based on the file
5. Show the expected output structure when done

Use these defaults unless the user specifies otherwise:
- FFmpeg: C:\Users\thiag\ffmpeg-full_build\bin\ffmpeg.exe
- Output: H:\Projects\GSplatTools\output
- Views: 4 (front/right/back/left)
- GPU: enabled (RTX 4090)
- Sharp frames: top 20%
- Skip masks unless user asks for them
