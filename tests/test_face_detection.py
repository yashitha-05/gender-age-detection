import cv2
from app.preprocessing import detect_faces, crop_face

img = cv2.imread("assets/sample1.jpg")
faces = detect_faces(img)

print(f"Detected {len(faces)} face(s)")

for f in faces:
    x, y, w, h = f['box']
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
