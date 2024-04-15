from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("D:/final_project/Anti-Spoofing/Anti-Spoofing/venv/l_version_1_300.pt")

classNames = ["fake", "real"]
prev_frame_time = 0
new_frame_time = 0

confidence_threshold = 0.5
confidence_threshold_real = 0.80

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    if not success:
        print("Error: Could not read a frame from the camera.")
        break

    results = model(img, stream=True, verbose=False)
    if results is not None:
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                print(conf)
                # Class Name
                cls = int(box.cls[0])

                if 0 <= cls < len(classNames):  # Check if the class index is valid
                    class_name = classNames[cls]
                    if conf >= confidence_threshold:
                        if class_name == "real" or conf >= confidence_threshold_real:
                            cvzone.putTextRect(img, f'{'Real'} {conf}', (max(
                                0, x1), max(35, y1)), scale=1, thickness=1)
                        elif class_name == "fake":
                            cvzone.putTextRect(img, f'{'Fake'} {conf}', (max(
                                0, x1), max(35, y1)), scale=1, thickness=1)
                else:
                    print(f"Invalid class index: {cls}")

    else:
        print("No results from the model.")

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    # print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
