import argparse
import os
import pickle
from pathlib import Path

import numpy as np

from runtime_env import ensure_runtime_paths

ensure_runtime_paths()

import cv2
import face_recognition

BASE_DIR = Path(__file__).resolve().parent
ENCODINGS_FILE = str(BASE_DIR / "encodings.pkl")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Use a generated demo frame when the camera is unavailable")
    parser.add_argument("--once", action="store_true", help="Process one frame and exit")
    return parser.parse_args()


def create_demo_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (20, 20, 20)
    cv2.rectangle(frame, (120, 120), (520, 360), (0, 255, 255), 2)
    cv2.circle(frame, (320, 240), 90, (255, 255, 255), 2)
    cv2.putText(frame, "Demo mode", (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "Camera unavailable", (170, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def load_encodings():
    if not os.path.exists(ENCODINGS_FILE):
        print("[WARNING] No encodings file found. Register people first!")
        return {"encodings": [], "names": []}

    with open(ENCODINGS_FILE, "rb") as f:
        return pickle.load(f)


def run_recognition(args):
    data = load_encodings()
    known_encodings = data["encodings"]
    known_names = data["names"]

    cap = None
    demo_mode = args.demo

    if not demo_mode:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[INFO] Camera could not be opened; falling back to demo mode.")
            demo_mode = True

    if demo_mode:
        print("[INFO] Starting demo mode. Press 'q' or run with --once to exit.")
    else:
        print("[INFO] Starting face recognition stream... Press 'q' to exit.")

    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Unable to read camera feed.")
                print("Please allow camera access in System Settings → Privacy & Security → Camera.")
                break
        else:
            frame = create_demo_frame()


        # Convert frame from BGR (OpenCV format) to RGB (face_recognition format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect all face locations and compute embeddings for the frame
        face_locations = face_recognition.face_locations(
            rgb_frame, model="hog"
        )
        face_encodings = face_recognition.face_encodings(
            rgb_frame, known_face_locations=face_locations
        )

        # Loop through each detected face in the frame
        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):
            name = "Unknown"
            color = (0, 0, 255)  # Red for unknown faces

            if known_encodings:
                # Compare face distance against saved face encodings
                matches = face_recognition.compare_faces(
                    known_encodings, face_encoding, tolerance=0.5
                )
                face_distances = face_recognition.face_distance(
                    known_encodings, face_encoding
                )

                if True in matches:
                    best_match_index = face_distances.argmin()
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        color = (0, 255, 0)  # Green for recognized faces

            # Draw bounding box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw label with name
            cv2.rectangle(
                frame,
                (left, bottom - 35),
                (right, bottom),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                name,
                (left + 6, bottom - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Display the output frame
        try:
            cv2.imshow("Face Detection & Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error:
            print("[INFO] Display window is unavailable; continuing in headless mode.")
            key = ord("q")

        if args.once or key == ord("q"):
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    run_recognition(args)