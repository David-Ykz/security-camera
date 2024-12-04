import cv2
import time

cap = cv2.VideoCapture(1)
NUM_PICTURES = 20
LABEL = "Unlocked"

for i in range(NUM_PICTURES):
    time.sleep(1)
    ret, frame = cap.read()
    
    cv2.imwrite(f"{LABEL}/{i+1}.jpg", frame)

cap.release()
