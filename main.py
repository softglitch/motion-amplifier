#!/usr/bin/env python3
import cv2
import numpy as np
import enum

GRAY = False

FRAME_BUFFER_SIZE = 5

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
    "motion_scale": 100,
    "heatmap_threshold": 40,
    "heatmap_color": 0,
    "heatmap_opacity": 0.5,
    "heamap_scale": 0.33,
}

WINDOW_APP = "Motion Amplifier"


def grayscale(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def denoise(frame: np.ndarray, blur_kernel: int = 5, scale: bool = False) -> np.ndarray:
    if scale:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = grayscale(frame)
    denoised_frame = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 1.2)
    if scale:
        denoised_frame = cv2.resize(denoised_frame, (frame.shape[1], frame.shape[0]))
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
    frame: np.ndarray, heatmap: np.ndarray, opacity: float = 0.5
) -> np.ndarray:
    overlay = cv2.addWeighted(frame, 1.0, heatmap, opacity, 0)
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

    render_text(frame, f"{current_frame} q to quit, 1,2,3..", line=0)
    render_text(frame, f"Heatmap: {Color(COLORS[CONFIG['heatmap_color']]).name}", line=1)
    render_text(frame, f"Opacity: {CONFIG['heatmap_opacity']}", line=2)
    render_text(frame, f"Gray: {GRAY}", line=3)
    cv2.imshow(WINDOW_APP, frame)


def add_motion(
    frame: np.ndarray,
    last_frame: np.ndarray,
    motion_scale: float = 2.0,
    gray: bool = False,
) -> np.ndarray:
    denoised_last_frame = denoise(last_frame)
    denoised_frame = denoise(frame)
    motion = get_motion(denoised_frame, denoised_last_frame)
    amplified_motion = cv2.multiply(motion, motion_scale)
    amplified_motion = cv2.normalize(amplified_motion, None, 0, 255, cv2.NORM_MINMAX)
    amplified_motion = amplified_motion.astype(np.uint8)
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
    overlay = create_overlay(frame, heatmap, opacity=CONFIG["heatmap_opacity"])
    return overlay


def main():
    pause = False
    cv2.namedWindow(WINDOW_APP, cv2.WINDOW_AUTOSIZE)
    # remove window decorations, but keep it resizable
    cv2.setWindowProperty(WINDOW_APP, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Open the MP4 file
    cap = cv2.VideoCapture("video.mp4")

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        print("Video file opened successfully!")

        last_frame = None
        last_frame_buffer = []
        _, last_frame = cap.read()
        current_frame = 0

        for _ in range(200):
            current_frame += 1
            _, last_frame = cap.read()
            last_frame_buffer.append(last_frame)

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

            # Arrow keys to skip forward/backward
            if key == 81:  # left arrow
                current_frame = max(0, current_frame - 30)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                _, last_frame = cap.read()
                continue
            if key == 83:  # right arrow
                for _ in range(30):
                    _, last_frame = cap.read()
                continue

            # space pauses the video
            if key == ord(" "):
                pause = not pause

            if not pause:
                current_frame += 1
                _, frame = cap.read()

                global GRAY
                if key == ord("3"):
                    GRAY = not GRAY
                overlay = add_motion(
                    frame, last_frame, motion_scale=CONFIG["motion_scale"], gray=GRAY
                )
                if len(last_frame_buffer) > FRAME_BUFFER_SIZE:
                    last_frame_buffer.pop(0)

                last_frame = frame.copy()
                last_frame_buffer.append(last_frame)
                render(overlay, current_frame)

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        print("Hello from motion-amplifier!")


if __name__ == "__main__":
    main()
