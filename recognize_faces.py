import os
import pickle
import cv2
import face_recognition

ENCODINGS_FILE = "encodings.pkl"


def load_encodings():
    if not os.path.exists(ENCODINGS_FILE):
        print("[WARNING] No encodings file found. Register people first!")
        return {"encodings": [], "names": []}

    with open(ENCODINGS_FILE, "rb") as f:
        return pickle.load(f)


def run_recognition():
    data = load_encodings()
    known_encodings = data["encodings"]
    known_names = data["names"]

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting face recognition stream... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Unable to access camera feed.")
            break

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
        cv2.imshow("Face Detection & Recognition", frame)

        # Exit stream on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_recognition()