#!/usr/bin/env python3
import cv2


def main():

    # Open the MP4 file
    cap = cv2.VideoCapture("video.mp4")

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        print("Video file opened successfully!")

        # Read and display frames
        # last_frame = None
        # print(frame)
        last_frame = None
        while True:
            _, frame = cap.read()
            original_frame = frame.copy()
            edges = cv2.Canny(frame, 100, 200)
            frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            copy_frame = frame.copy()

            if last_frame is None:
                last_frame = frame
                continue

            # Get difference between current frame and last frame to amplify motion
            delta = cv2.absdiff(frame, last_frame)
            amplification = cv2.multiply(delta, (0, 0, 255))  # amplify red channel to make motion more visible
            last_frame = copy_frame

            # over lay delta on original frame
            frame = cv2.addWeighted(original_frame, 1, amplification, 1, 0)
            # draw frame
            cv2.imshow("Video", frame)


            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        #     ret, frame = cap.read()
        #     if not ret:
        #         print("End of video or error occurred.")
        #         break
        #     cv2.imshow("Video", frame)
        #     if cv2.waitKey(25) & 0xFF == ord("q"):
        #         break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        print("Hello from motion-amplifier!")


if __name__ == "__main__":
    main()
