# Motion Amplifier

Realtime motion-visualization tool built with OpenCV.
It overlays a configurable heatmap on top of a video to make subtle movement easier to see.

## Requirements

- Python 3.12+
- OpenCV (installed via project dependencies)

## Install

```bash
uv sync
```

## Run

```bash
python3 main.py
```

Use a custom video path:

```bash
python3 main.py /path/to/video.mp4
```

## Controls

- `q`: Quit
- `space`: Pause/unpause
- `r`: Restart from beginning
- `left arrow`: Jump backward ~30 frames
- `right arrow`: Jump forward ~30 frames
- `1`: Cycle heatmap color
- `2`: Cycle heatmap opacity
- `3`: Toggle grayscale mode
- `4`: Increase motion amplification scale (loops back after max)
- `5`: Toggle source image visibility
