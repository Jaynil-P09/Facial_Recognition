import cv2
import numpy as np
from deepface import DeepFace

# Load known faces with multiple embeddings per person
known_faces = np.load('known_faces.npy', allow_pickle=True).item()

# Use Haar Cascade for face detection (or switch to DNN for better speed)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Comparison threshold
threshold = 3.0

# Simplified face comparison using vectorized Euclidean distance
def compare_faces(known_embeddings, unknown_embedding):
    # Convert the list of known embeddings to a numpy array for efficient computation
    known_embeddings = np.array(known_embeddings)

    # Calculate the Euclidean distances between the unknown embedding and all known embeddings
    distances = np.linalg.norm(known_embeddings - unknown_embedding, axis=1)

    # Return the minimum distance
    return np.min(distances)


# Function to process the face crop
def process_face(face_crop, known_faces):
    try:
        # Ensure DeepFace uses GPU
        result = DeepFace.represent(face_crop, model_name="ArcFace", enforce_detection=False)
        unknown_embedding = np.array(result[0]['embedding'])

        # Find the best match in known faces using the simplified comparison
        best_match = "Unknown"
        min_distance = float('inf')

        # Loop through the known faces to find the best match
        for name, embeddings in known_faces.items():
            distance = compare_faces(embeddings, unknown_embedding)
            if distance < min_distance:
                min_distance = distance
                best_match = name

        if min_distance > threshold:
            best_match = "Unknown"

        return best_match
    except Exception as e:
        print(f"Error processing face: {e}")
        return "Unknown"


# Start webcam/video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to reduce processing time
    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y + h, x:x + w]

        # Process the face and get the best match
        best_match = process_face(face_crop, known_faces)

        # Draw rectangle and name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{best_match}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame with overlay
    cv2.imshow('Video', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()




