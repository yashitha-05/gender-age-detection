import cv2
import math
import numpy as np
import os

# Paths to model files
faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"

ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"

genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Labels
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-30)', '(31-40)', '(41-60)', '(61-100)']
genderList = ['Male', 'Female']

# Model mean values
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# --- Helper Functions ---
def getFaceBox(net, frame, conf_threshold=0.7):
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
    return bboxes

def get_contrasting_color(bgr_color):
    """Ensure text color contrasts with background rectangle"""
    brightness = (bgr_color[0]*0.299 + bgr_color[1]*0.587 + bgr_color[2]*0.114)
    return (0, 0, 0) if brightness > 150 else (255, 255, 255)

# --- Webcam Detection ---
cap = cv2.VideoCapture(0)
padding = 20
print("🎥 Starting real-time gender and age detection... Press 'q' to quit.")

prev_gender, prev_age = [], []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frameHeight, frameWidth = frame.shape[:2]
    bboxes = getFaceBox(faceNet, frame)

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        face = frame[max(0, y1 - padding):min(y2 + padding, frameHeight - 1),
                     max(0, x1 - padding):min(x2 + padding, frameWidth - 1)]

        # Blob for networks
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Predict gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        # Predict age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()

        # --- Smoothing predictions ---
        prev_gender.append(genderPreds[0])
        prev_age.append(agePreds[0])
        if len(prev_gender) > 10:
            prev_gender.pop(0)
            prev_age.pop(0)

        avg_gender = np.mean(prev_gender, axis=0)
        avg_age = np.mean(prev_age, axis=0)

        gender = genderList[avg_gender.argmax()]
        age = ageList[avg_age.argmax()]
        gender_conf = avg_gender.max()
        age_conf = avg_age.max()

        # --- Draw rectangle around face ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 2)

        # --- Auto-contrast text ---
        color_rect = (0, 255, 0)
        text_color = get_contrasting_color(color_rect)

        label = f"{gender}, {age}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

        # --- Confidence Bar ---
        bar_x, bar_y = x1, y2 + 30
        bar_w, bar_h = int(200 * gender_conf), 15
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 200, bar_y + bar_h), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 255, 0), -1)
        cv2.putText(frame, f"{gender_conf*100:.1f}%", (bar_x + 210, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # --- Auto-save confident detections ---
        if gender_conf > 0.85 and age_conf > 0.85:
            os.makedirs("captures", exist_ok=True)
            filename = f"captures/{gender}_{age}_{int(gender_conf*100)}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Saved snapshot: {filename}")

    # ✅ Added “Press Q to quit” message on screen
    cv2.putText(frame, "Press 'Q' to quit", (20, frameHeight - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Gender and Age Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Stopped.")
