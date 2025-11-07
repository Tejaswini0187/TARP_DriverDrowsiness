import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
from tqdm import tqdm

# CONFIG
VIDEOS_DIR = "C:/Users/motak/OneDrive/Desktop/TARP/Videos"      # input videos
DATASET_DIR = "C:/Users/motak/OneDrive/Desktop/TARP/dataset"    # output dataset folder
OPEN_DIR = os.path.join(DATASET_DIR, "open")
CLOSED_DIR = os.path.join(DATASET_DIR, "closed")
FACE_PREDICTOR_PATH = "C:/Users/motak/OneDrive/Desktop/TARP/shape_predictor_68_face_landmarks.dat"

FRAME_SKIP = 5        # save every 5th frame
EAR_THRESH = 0.25     # threshold to classify closed eyes
FACE_SIZE = (128, 128)  # resized face image size

# Helper Functions
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Create directories
ensure_dir(OPEN_DIR)
ensure_dir(CLOSED_DIR)

# Initialize dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FACE_PREDICTOR_PATH)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Process videos
videos = [v for v in os.listdir(VIDEOS_DIR) if v.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

open_count = 1
closed_count = 1

for video_name in videos:
    video_path = os.path.join(VIDEOS_DIR, video_name)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    pbar = tqdm(total=frame_count//FRAME_SKIP, desc=f"Processing {video_name}", unit="frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)

            for face in faces:
                # Full face crop
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                x1, y1, x2, y2 = max(0, x-10), max(0, y-10), min(frame.shape[1], x+w+10), min(frame.shape[0], y+h+10)
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                face_crop = cv2.resize(face_crop, FACE_SIZE)
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

                # Eye aspect ratio for open/closed labeling
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

                # Save labeled frame
                if ear < EAR_THRESH:
                    out_path = os.path.join(CLOSED_DIR, f"closed{closed_count}.jpg")
                    cv2.imwrite(out_path, gray_face)
                    closed_count += 1
                else:
                    out_path = os.path.join(OPEN_DIR, f"open{open_count}.jpg")
                    cv2.imwrite(out_path, gray_face)
                    open_count += 1

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

print(f"\n✅ Done! Saved {open_count-1} open-face frames and {closed_count-1} closed-face frames.")

