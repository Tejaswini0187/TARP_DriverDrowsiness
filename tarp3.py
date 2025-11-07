from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame
import time

# CONFIGURATION
MODEL_PATH = "C:/Users/motak/OneDrive/Desktop/TARP/eye_state_cnn.h5"
PREDICTOR_PATH = "C:/Users/motak/OneDrive/Desktop/TARP/shape_predictor_68_face_landmarks.dat"
ALERT_SOUND = "C:/Users/motak/OneDrive/Desktop/TARP/music.wav"

# Thresholds 
EAR_THRESH = 0.30        # higher for easier triggering
CNN_THRESH = 0.6         # lower for easier triggering
CLOSED_FRAME_LIMIT = 15  # shorter duration for testing
FACE_SIZE = (128, 128)

# INITIALIZE MIXER FOR ALERT SOUND
pygame.mixer.init()
pygame.mixer.music.load(ALERT_SOUND)
pygame.mixer.music.set_volume(1.0)

# LOAD CNN MODEL AND DLIB MODELS
print("[INFO] Loading CNN model and Dlib predictor...")
model = load_model(MODEL_PATH)
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(PREDICTOR_PATH)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# HELPER FUNCTION: EYE ASPECT RATIO
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# VIDEO CAPTURE
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
fps = 30 if fps == 0 else fps

EYE_FRAME_COUNT = 0
ALERT_ACTIVE = False

print("[INFO] Starting detection... Press 'Q' to quit.")

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract eye landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # Extract face region for CNN
        x, y, w, h = subject.left(), subject.top(), subject.width(), subject.height()
        x1, y1 = max(0, x - 10), max(0, y - 10)
        x2, y2 = min(frame.shape[1], x + w + 10), min(frame.shape[0], y + h + 10)
        face_crop = gray[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        face_crop = cv2.resize(face_crop, FACE_SIZE).astype("float32") / 255.0
        face_crop = np.expand_dims(face_crop, axis=(0, -1))

        # CNN prediction (0=open, 1=closed)
        pred = model.predict(face_crop)[0][0]

        # Combine EAR + CNN
        if ear < EAR_THRESH and pred > CNN_THRESH:
            EYE_FRAME_COUNT += 1
        else:
            EYE_FRAME_COUNT = 0

        # Drowsiness alert logic
        if EYE_FRAME_COUNT >= CLOSED_FRAME_LIMIT:
            cv2.putText(frame, "******** DROWSINESS ALERT ********", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            if not pygame.mixer.music.get_busy():   # Play alert if not already playing
                print("[DEBUG] ALERT triggered!")   
                pygame.mixer.music.play(-1)       
                ALERT_ACTIVE = True
        else:
            if ALERT_ACTIVE:      
                pygame.mixer.music.stop()
                ALERT_ACTIVE = False

    # Show video feed
    cv2.imshow("Drowsiness Detection (EAR + CNN)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
# CLEANUP
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("✅ Detection stopped successfully.")
