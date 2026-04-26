Add a new camera view entry to a JSON config file.

Arguments: $ARGUMENTS (format: "name yaw pitch [config_path]")
Example: /add-view drone_low -30 45 configs/cameras_drone_orbit.json

Steps:
1. Parse name, yaw, pitch from arguments (roll defaults to 0)
2. If config_path given, read the file and append the new view to the "views" array
3. If no config_path, show the JSON snippet to add manually
4. Validate that yaw is in [-180, 180] (FFmpeg v360 constraint) — warn and correct if not
5. Write the updated file and confirm

The view object format: {"name": "...", "yaw": N, "pitch": N, "roll": 0}
