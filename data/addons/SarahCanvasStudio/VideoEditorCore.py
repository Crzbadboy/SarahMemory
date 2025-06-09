# VideoEditorCore.py
# Basic AI-driven video editor module for SarahMemory

import cv2
import os
import datetime

VIDEO_IN_DIR = os.path.join("data", "videos")
VIDEO_OUT_DIR = os.path.join("data", "video_output")

if not os.path.exists(VIDEO_OUT_DIR):
    os.makedirs(VIDEO_OUT_DIR)


def cut_video(input_file, start_sec, end_sec):
    input_path = os.path.join(VIDEO_IN_DIR, input_file)
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("[VideoEditor] Error: Cannot open video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_name = f"cut_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    output_path = os.path.join(VIDEO_OUT_DIR, output_name)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame >= start_frame and current_frame <= end_frame:
            out.write(frame)
        current_frame += 1
        if current_frame > end_frame:
            break

    cap.release()
    out.release()
    print(f"[VideoEditor] Cut video saved to {output_path}")


def overlay_text(input_file, output_name, text):
    input_path = os.path.join(VIDEO_IN_DIR, input_file)
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("[VideoEditor] Error: Cannot open video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = os.path.join(VIDEO_OUT_DIR, output_name)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)

    cap.release()
    out.release()
    print(f"[VideoEditor] Overlay video saved to {output_path}")


if __name__ == '__main__':
    overlay_text("sample.mp4", "overlay_demo.mp4", "Sarah was here")
