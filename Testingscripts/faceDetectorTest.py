from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone

cap = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
    success, img = cap.read()

    if not success:
        print("Error: Could not read frame from the camera.")
        break

    img, bboxs = detector.findFaces(img)

    if bboxs:
        center = bboxs[0]["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
