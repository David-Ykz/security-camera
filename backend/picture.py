import cv2

# Initialize the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
else:
    # Read a single frame from the webcam
    ret, frame = cap.read()

    if ret:
        # Save the captured frame as 'recognizedFace1.jpg'
        cv2.imwrite('recognizedFace1.jpg', frame)
        print("Image saved as recognizedFace1.jpg")
    else:
        print("Error: Could not capture the frame.")

    # Release the webcam
    cap.release()

# Close any OpenCV windows (if any)
cv2.destroyAllWindows()
