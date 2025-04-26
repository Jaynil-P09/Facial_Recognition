import cv2 as cv
import os

# Set the person's name or label for the folder
person_name = "Viduna"  # Change this to the person's name or ID
os.makedirs(f"data/{person_name}", exist_ok=True)

# Load Haar cascade for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)
index = 0
frame_counter = 0  # To capture every third frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        if frame_counter % 3 == 0:
            image_name = f"data/{person_name}/{person_name}_{index}.jpg"
            cv.imwrite(image_name, face_img)
            print(f"Image saved as {image_name}")

            if index == 300:
                print("300 images saved!")
            elif index == 500:
                print("500 images saved!")

            index += 1

        # Draw a rectangle around the face for display
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    frame_counter += 1

    # Display the frame
    cv.imshow("Face Capture", frame)

    # Exit if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
