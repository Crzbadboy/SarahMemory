# SarahVideoCore.py
# Handles AI video decoding, analysis, and overlay display for real-time interaction

import cv2
import os
import time
import numpy as np

VIDEO_INPUT_PATH = os.path.join("data", "videos")
VIDEO_OUTPUT_PATH = os.path.join("data", "video_output")

if not os.path.exists(VIDEO_OUTPUT_PATH):
    os.makedirs(VIDEO_OUTPUT_PATH)


def analyze_and_overlay(video_file, display=False):
    input_path = os.path.join(VIDEO_INPUT_PATH, video_file)
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"[VideoCore] Error: Cannot open video file {video_file}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_file = os.path.join(VIDEO_OUTPUT_PATH, f"overlay_{video_file}")
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        overlay_text = f"Frame {frame_id}"
        cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)
        if display:
            cv2.imshow("SarahVideoCore Output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[VideoCore] Completed overlay output: {output_file}")


if __name__ == '__main__':
    test_file = "sample.mp4"
    analyze_and_overlay(test_file, display=True)