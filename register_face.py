import os
import pickle
import cv2
import face_recognition

DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pkl"


def load_existing_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}


def register_new_person(person_name, max_samples=5):
    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    data = load_existing_encodings()

    cap = cv2.VideoCapture(0)
    print(
        f"[INFO] Starting webcam for '{person_name}'. Press 'SPACE' to capture face, or 'q' to quit."
    )

    count = 0
    while count < max_samples:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to access webcam.")
            break

        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            f"Captured: {count}/{max_samples} (Press SPACE)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Register Face", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):  # Space key pressed
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb_frame, model="hog")

            if len(boxes) == 1:
                # Save crop image
                img_path = os.path.join(person_dir, f"{count + 1}.jpg")
                cv2.imwrite(img_path, frame)

                # Compute facial encoding
                encoding = face_recognition.face_encodings(
                    rgb_frame, known_face_locations=boxes
                )[0]
                data["encodings"].append(encoding)
                data["names"].append(person_name)

                count += 1
                print(f"[SUCCESS] Captured photo {count}/{max_samples}")
            elif len(boxes) == 0:
                print("[WARNING] No face detected. Please position properly.")
            else:
                print(
                    "[WARNING] Multiple faces detected! Ensure only one person is in frame."
                )

        elif key == ord("q"):
            print("[INFO] Registration canceled.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if count > 0:
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)
        print(
            f"[INFO] Successfully registered '{person_name}' with {count} face embeddings!"
        )


if __name__ == "__main__":
    name = input("Enter the full name of the person to register: ").strip()
    if name:
        register_new_person(name)
    else:
        print("[ERROR] Name cannot be empty.")