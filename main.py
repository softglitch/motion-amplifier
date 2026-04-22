#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import enum

GRAY = False

FRAME_BUFFER_SIZE = 5

SCALE = 1.0


class Color(enum.Enum):
    RED = [0, 0, 255]
    GREEN = [0, 255, 0]
    BLUE = [255, 0, 0]
    YELLOW = [0, 255, 255]
    CYAN = [255, 255, 0]
    MAGENTA = [255, 0, 255]


COLORS = [
    Color.RED.value,
    Color.GREEN.value,
    Color.BLUE.value,
    Color.CYAN.value,
    Color.YELLOW.value,
    Color.MAGENTA.value,
]

CONFIG = {
    "blur_kernel": 5,
    "blur": True,
    "motion_scale": 1,
    "heatmap_threshold": 40,
    "heatmap_color": 0,
    "heatmap_opacity": 0.5,
    "heamap_scale": 0.33,
    "opacity": 1.0,
    "edges": False,
}

WINDOW_APP = "Motion Amplifier"


def grayscale(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def denoise(frame: np.ndarray, blur_kernel: int = 5) -> np.ndarray:
    gray = grayscale(frame)
    denoised_frame = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 1.2)
    return denoised_frame.astype(np.float32)


def get_motion(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    # compute difference in motion
    motion = cv2.absdiff(frame1, frame2)
    return motion


def create_heatmap(
    motion: np.ndarray,
    threshold: int = 30,
    color: list = [0, 0, 255],
    scale: float = 0.33,
) -> np.ndarray:
    # Set heat map color based on motion intensity
    _, motion_mask = cv2.threshold(
        motion.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY
    )
    heatmap = cv2.applyColorMap(motion_mask, 1)
    # scale heatmap down to 25% size for speed
    heatmap = cv2.resize(heatmap, (0, 0), fx=scale, fy=scale)
    heatmap[np.where((heatmap == [255, 255, 255]).all(axis=2))] = color
    heatmap = cv2.resize(heatmap, (motion.shape[1], motion.shape[0]))
    return heatmap


def create_overlay(
    frame: np.ndarray,
    heatmap: np.ndarray,
    opacity: float = 1.0,
    heatmap_opacity: float = 0.5,
) -> np.ndarray:
    overlay = cv2.addWeighted(frame, opacity, heatmap, heatmap_opacity, 0)
    return overlay


def render(frame: np.ndarray, current_frame: int) -> None:
    # Add text overlay with instructions
    # Font should be small and in the bottom left corner, with white text and a black outline for visibility

    def render_text(frame: np.ndarray, text: str, line: int = 0) -> None:
        # top left corner of the text should be 10 pixels from the bottom and 10 pixels from the left
        font = cv2.FONT_HERSHEY_COMPLEX
        stroke = (255, 255, 255)
        color = (0, 0, 0)
        size = 0.5
        x = 10
        y = 30 * (line + 1) * size
        pos = (x, int(y))
        font_size = 1
        stroke_size = 4
        cv2.putText(
            frame, text, pos, font, size, color, font_size * stroke_size, cv2.LINE_AA
        )
        cv2.putText(frame, text, pos, font, size, stroke, font_size, cv2.LINE_AA)

    render_text(frame, f"{current_frame} q to quit, options: 1,2,3..", line=0)
    render_text(
        frame, f"1:Heatmap: {Color(COLORS[CONFIG['heatmap_color']]).name}", line=1
    )
    render_text(frame, f"2:Opacity: {CONFIG['heatmap_opacity']}", line=2)
    render_text(frame, f"3:Gray: {GRAY}", line=3)
    render_text(frame, f"4:Motion Scale: {CONFIG['motion_scale']}", line=4)
    render_text(frame, f"5:Image: {'On' if CONFIG['opacity'] > 0 else 'Off'}", line=5)
    render_text(frame, f"6:Blur Kernel: {CONFIG['blur_kernel']}", line=6)
    render_text(frame, f"7:Blur Enabled: {CONFIG['blur']}", line=7)
    render_text(frame, f"8:Edges Enabled: {CONFIG['edges']}", line=8)
    cv2.imshow(WINDOW_APP, frame)


def add_motion(
    frame: np.ndarray,
    last_frame: np.ndarray,
    motion_scale: int = 2,
    gray: bool = False,
) -> np.ndarray:
    if SCALE != 1.0:
        frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        last_frame = cv2.resize(last_frame, (0, 0), fx=SCALE, fy=SCALE)

    if CONFIG["blur"]:
        blur_kernel = CONFIG["blur_kernel"]
        denoised_last_frame = denoise(last_frame, blur_kernel=blur_kernel)
        denoised_frame = denoise(frame, blur_kernel=blur_kernel)
    else:
        denoised_last_frame = grayscale(last_frame)
        denoised_frame = grayscale(frame)
    motion = get_motion(denoised_last_frame, denoised_frame)
    amplified_motion = cv2.dilate(
        motion.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=motion_scale
    )
    color = COLORS[CONFIG["heatmap_color"]]
    heatmap = create_heatmap(
        amplified_motion,
        threshold=CONFIG["heatmap_threshold"],
        color=color,
        scale=CONFIG["heamap_scale"],
    )
    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    overlay = create_overlay(
        frame,
        heatmap,
        opacity=CONFIG["opacity"],
        heatmap_opacity=CONFIG["heatmap_opacity"],
    )
    if CONFIG["edges"]:
        # highlight differences in edges between frames
        edges1 = cv2.Canny(denoised_last_frame.astype(np.uint8), 100, 200)
        edges2 = cv2.Canny(denoised_frame.astype(np.uint8), 100, 200)
        edge_diff = cv2.absdiff(edges1, edges2)
        edge_overlay = cv2.cvtColor(edge_diff, cv2.COLOR_GRAY2BGR)
        edge_overlay[np.where((edge_overlay == [255, 255, 255]).all(axis=2))] = [
            0,
            255,
            255,
        ]
        edge_overlay = cv2.GaussianBlur(edge_overlay, (3, 3), 0)
        overlay = cv2.addWeighted(overlay, 1.0, edge_overlay, 0.5, 0)

    return overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Amplify motion in a video and display a heatmap overlay."
    )
    parser.add_argument(
        "video",
        help="Path to input video file.",
    )
    parser.add_argument(
        "--skip-frames",
        help="Number of initial frames to skip.",
        type=int,
    )
    parser.add_argument(
        "-s",
        "--scale",
        help="Scale final output (0.5 = 50% size).",
        type=float,
        default=1.0,
    )
    return parser.parse_args()


def main():
    global GRAY
    global SCALE
    args = parse_args()
    SCALE = args.scale
    pause = False
    cv2.namedWindow(WINDOW_APP, cv2.WINDOW_AUTOSIZE)
    # remove window decorations, but keep it resizable
    cv2.setWindowProperty(WINDOW_APP, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Open the input video file
    cap = cv2.VideoCapture(args.video)

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file: {args.video}")
    else:
        print("Video file opened successfully!")

        even_older_frame = None
        last_frame = None
        ok, last_frame = cap.read()
        if not ok or last_frame is None:
            print(f"Error: Could not read first frame from video file: {args.video}")
            cap.release()
            cv2.destroyAllWindows()
            return
        current_frame = 0

        if args.skip_frames:
            for _ in range(args.skip_frames):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                current_frame += 1
                even_older_frame = last_frame.copy()
                last_frame = frame.copy()

        while True:
            # handle key events
            key = cv2.waitKey(25) & 0xFF
            if key == ord("q"):
                break
            if key == ord("1"):
                current_color = CONFIG["heatmap_color"]
                CONFIG["heatmap_color"] = (current_color + 1) % len(COLORS)
            if key == ord("2"):
                current_opacity = CONFIG["heatmap_opacity"]
                if current_opacity == 0.5:
                    CONFIG["heatmap_opacity"] = 0.75
                elif current_opacity == 0.75:
                    CONFIG["heatmap_opacity"] = 1.0
                elif current_opacity == 1.0:
                    CONFIG["heatmap_opacity"] = 0.0
                elif current_opacity == 0.0:
                    CONFIG["heatmap_opacity"] = 0.25
                else:
                    CONFIG["heatmap_opacity"] = 0.5

            if key == ord("r"):
                # reset video to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_frame = 0
                ok, first_frame = cap.read()
                if not ok or first_frame is None:
                    print("Reached end of video or failed to decode after restart.")
                    break
                last_frame = first_frame.copy()
                even_older_frame = None
                if pause:
                    overlay = add_motion(
                        last_frame,
                        last_frame,
                        motion_scale=CONFIG["motion_scale"],
                        gray=GRAY,
                    )
                    render(overlay, current_frame)
                continue

            # Arrow keys to skip forward/backward
            if key == 81:  # left arrow
                current_frame = max(0, current_frame - 30)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ok, frame = cap.read()
                if not ok or frame is None:
                    print(
                        "Reached end of video or failed to decode while seeking backward."
                    )
                    break
                last_frame = frame.copy()
                even_older_frame = None
                if pause:
                    overlay = add_motion(
                        last_frame,
                        last_frame,
                        motion_scale=CONFIG["motion_scale"],
                        gray=GRAY,
                    )
                    render(overlay, current_frame)
                continue
            if key == 83:  # right arrow
                skipped = 0
                for _ in range(30):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    skipped += 1
                    last_frame = frame.copy()
                current_frame += skipped
                if skipped == 0:
                    print(
                        "Reached end of video or failed to decode while seeking forward."
                    )
                    break
                even_older_frame = None
                if pause:
                    overlay = add_motion(
                        last_frame,
                        last_frame,
                        motion_scale=CONFIG["motion_scale"],
                        gray=GRAY,
                    )
                    render(overlay, current_frame)
                continue

            # space pauses the video
            if key == ord(" "):
                pause = not pause

            # set motion scale
            if key == ord("4"):
                current_scale = CONFIG["motion_scale"]
                if current_scale >= 5:
                    CONFIG["motion_scale"] = 1
                else:
                    CONFIG["motion_scale"] = current_scale + 1

            # Toggle image
            if key == ord("5"):
                CONFIG["opacity"] = 0.0 if CONFIG["opacity"] > 0 else 1.0

            if key == ord("6"):
                current_blur = CONFIG["blur_kernel"]
                if current_blur >= 15:
                    CONFIG["blur_kernel"] = 1
                else:
                    CONFIG["blur_kernel"] = current_blur + 2

            if key == ord("7"):
                CONFIG["blur"] = not CONFIG["blur"]

            if key == ord("8"):
                CONFIG["edges"] = not CONFIG["edges"]

            if not pause:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Reached end of video or failed to decode frame.")
                    break
                current_frame += 1
            else:
                frame = last_frame.copy()
                if even_older_frame is not None:
                    last_frame = even_older_frame.copy()

            if key == ord("3"):
                GRAY = not GRAY
            overlay = add_motion(
                frame, last_frame, motion_scale=CONFIG["motion_scale"], gray=GRAY
            )
            even_older_frame = last_frame.copy()
            last_frame = frame.copy()
            render(overlay, current_frame)

        # Release resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
