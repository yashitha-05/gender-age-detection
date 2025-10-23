import cv2
from app.preprocessing import detect_faces

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("📷 Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    # Detect faces in the current frame
    faces = detect_faces(frame)

    # Draw bounding boxes
    for f in faces:
        x, y, w, h = f['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Real-Time Face Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
