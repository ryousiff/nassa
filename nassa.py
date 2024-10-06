import cv2
import mediapipe as mp
from fer import FER


# Initialize the camera (one-time initialization)
cap = cv2.VideoCapture(0)

# Load pre-trained face detector (ensure correct path)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize Mediapipe's Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Initialize emotion detector (once)
emotion_detector = FER()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar cascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convert to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for face landmarks
    result = face_mesh.process(rgb_frame)
    
    # Draw face landmarks if detected
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for id, lm in enumerate(face_landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Detect emotion in the frame using FER
    emotion, score = emotion_detector.top_emotion(frame)
    if emotion:
        cv2.putText(frame, f"Emotion: {emotion}, Score: {score:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the combined output in the window
    cv2.imshow('AI Detector (Face, Landmarks, Emotion)', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows properly
cap.release()
cv2.destroyAllWindows()
