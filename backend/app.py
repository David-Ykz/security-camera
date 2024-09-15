from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import cv2
from datetime import datetime, timedelta
import numpy as np
import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv

load_dotenv()
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')
SERVER_EMAIL = os.getenv('SERVER_EMAIL')
SERVER_PASSWORD = os.getenv('SERVER_PASSWORD')
AUTHENTICATION_TOKEN = os.getenv('AUTHENTICATION_TOKEN')
VIDEO_KEY = os.getenv('VIDEO_KEY')

MOTION_THRESHOLD = 0.9
MOTION_ALERT_LIMIT = 25
SEND_EMAIL_COOLDOWN = timedelta(minutes=5)
SEND_EMAIL_FLAG = False

app = Flask(__name__)
CORS(app)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
camera = cv2.VideoCapture(1)

def prepareTrainingData(imagePath):
    grayImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    faces = faceCascade.detectMultiScale(grayImage, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        print("No face found in the reference image.")
        return None, None
    (x, y, w, h) = faces[0]
    return grayImage[y:y + h, x:x + w], 1  # Label '1' for the known person

def detectAndRecognizeFace(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Recognize the face (if recognizer is trained)
        faceROI = gray[y:y + h, x:x + w]
        label, confidence = faceRecognizer.predict(faceROI)
        if label == 1:
            text = f"Recognized (Confidence: {confidence:.2f})"
        else:
            text = "Unknown"
        
        # Display recognition result
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

def sendEmail():
    msg = EmailMessage()
    msg.set_content('Your camera has detected motion.')
    msg['Subject'] = 'Motion detected'
    msg['From'] = SERVER_EMAIL
    msg['To'] = RECIPIENT_EMAIL

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SERVER_EMAIL, SERVER_PASSWORD)
            server.send_message(msg)
            print('Email sent successfully!')
    except Exception as e:
        print(f'Error sending email: {e}')

def detectMotion(prevMeanBrightness, frame):
    grayscaleImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    currMeanBrightness = np.mean(grayscaleImage)
    if np.abs(currMeanBrightness - prevMeanBrightness) > MOTION_THRESHOLD:
        print(f"{datetime.now()}: Motion detected, difference value: {currMeanBrightness - prevMeanBrightness}")
        return True, currMeanBrightness

    return False, currMeanBrightness


trainingImagePath = 'recognizedFace1.jpg'
trainingImage, label = prepareTrainingData(trainingImagePath)
if trainingImage is not None:
    faceRecognizer.train([trainingImage], np.array([label]))


def generate_frames():
    prevMeanBrightness = 0
    motionCounter = 0
    
    prevEmailSentTime = datetime.now() - SEND_EMAIL_COOLDOWN

    while True:
        success, frame = camera.read()
        if not success:
            print("failed")
            break
        else:
            detectedMotion, prevMeanBrightness = detectMotion(prevMeanBrightness, frame)
            if detectedMotion:
                motionCounter += 1
                if motionCounter > MOTION_ALERT_LIMIT and SEND_EMAIL_FLAG:
                    if datetime.now() - prevEmailSentTime > SEND_EMAIL_COOLDOWN:
                        prevEmailSentTime = datetime.now()
                        sendEmail()
            elif motionCounter > 0:
                motionCounter -= 1

            # Transmit frame to frontend
            frame = detectAndRecognizeFace(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route(f'/{VIDEO_KEY}')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_feed', methods=['POST'])
# def video_feed():
#     data = request.get_json()
#     print(data)
#     token = data.get('token')
#     if token == AUTHENTICATION_TOKEN:
#         return Response(generate_frames(),
#                         mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/authenticate', methods=['POST'])
def test():
    data = request.get_json()
    token = data.get('token')
    print(token)
    if token == AUTHENTICATION_TOKEN:
        response = {
            "videoKey": f"http://192.168.50.179:5000/{VIDEO_KEY}",
            "display": "true"
        }
    else:
        response = {
            "videoKey": "",
            "display": "false"
        }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# import ctypes
# import cv2
# import numpy as np
# from flask import Flask, Response

# app = Flask(__name__)


# # WARNING: The scripts f2py, flask, and numpy-config are installed in '/home/orangepi/.local/bin' which is not on PATH.


# # Load the shared library (compiled C++ code)
# video_capture = ctypes.CDLL('./video_capture.so')

# # Define the C++ function signature (return types)
# video_capture.capture_frame.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_int)]
# video_capture.capture_frame.restype = ctypes.c_int

# def capture_frame():
#     # Initialize pointers for the byte array and size
#     data_ptr = ctypes.POINTER(ctypes.c_ubyte)()
#     size = ctypes.c_int()

#     # Call the C++ function
#     if video_capture.capture_frame(ctypes.byref(data_ptr), ctypes.byref(size)) == 0:
#         # Convert the byte array back to a numpy array (which OpenCV uses)
#         byte_array = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_ubyte * size.value)).contents
#         np_array = np.frombuffer(byte_array, dtype=np.uint8)

#         # Decode the JPEG image to a format Flask can stream (OpenCV Mat format)
#         frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

#         # Free the memory allocated by C++ (if applicable)
# #        ctypes.free(data_ptr)

#         # Encode frame back to JPEG format for streaming
#         _, jpeg = cv2.imencode('.jpg', frame)
#         return jpeg.tobytes()
#     else:
#         return None

# @app.route('/video_feed')
# def video_feed():
#     def generate():
#         while True:
#             frame = capture_frame()
#             if frame:
#                 # Streaming MJPEG format
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
