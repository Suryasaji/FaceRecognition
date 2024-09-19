import cv2
import face_recognition
import os
import numpy as np

# Load known faces from a folder
def load_known_faces(folder_path):
    known_faces = []
    known_labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = face_recognition.load_image_file(os.path.join(folder_path, filename))
            image_np = np.array(image)
            encodings = face_recognition.face_encodings(image_np)
            if encodings:  # Only add if a face encoding was found
                known_faces.append(encodings[0])
                known_labels.append(filename.split('.')[0])  # Use filename (without extension) as label
    return known_faces, known_labels

# Process video and compare detected faces with known faces
def process_video_frames(known_faces, known_labels):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame from BGR (OpenCV default) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using face_recognition
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare the detected face with known faces
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
            label = "Unknown"
            if True in matches:
                match_index = matches.index(True)
                label = known_labels[match_index]
            
            # Draw rectangle and label on the frame
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Example Usage
if __name__ == "__main__":
    folder_path = "C:\\Users\\Surya\\face_recognition\\known_faces"  # Replace with the path to your known faces folder
    known_faces, known_labels = load_known_faces(folder_path)
    process_video_frames(known_faces, known_labels)
