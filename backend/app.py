from flask import Flask, Response
from flask_cors import CORS
import cv2
import datetime
import numpy as np
#import face_recognition

app = Flask(__name__)
CORS(app)

camera = cv2.VideoCapture(1)
MOTION_THRESHOLD = 0.9

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

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


trainingImagePath = 'recognizedFace1.jpg'
trainingImage, label = prepareTrainingData(trainingImagePath)
if trainingImage is not None:
    faceRecognizer.train([trainingImage], np.array([label]))



# recognizedFace1 = face_recognition.load_image_file("recognizedFace1.jpg")
# recognizedFace1Encoding = face_recognition.face_encodings(recognizedFace1)[0]

# knownFaces = [recognizedFace1Encoding]

def generate_frames():
    prevMeanBrightness = 0
    while True:
        success, frame = camera.read()
        if not success:
            print("failed")
            break
        else:
            grayscaleImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            currMeanBrightness = np.mean(grayscaleImage)
            if np.abs(currMeanBrightness - prevMeanBrightness) > MOTION_THRESHOLD:
                print(f"{datetime.datetime.now()}: Motion detected, difference value: {currMeanBrightness - prevMeanBrightness}")
            prevMeanBrightness = currMeanBrightness



            # rgbFrame = frame[:, :, ::-1]
            # faceLocations = face_recognition.face_locations(rgbFrame)
            # faceEncodings = face_recognition.face_encodings(rgbFrame, faceLocations)
            # faceNames = []
            # for face_encoding in faceEncodings:
            #     matches = face_recognition.compare_faces(knownFaces, faceEncodings)
            #     name = "Unknown"

            #     # # If a match was found in known_face_encodings, just use the first one.
            #     # if True in matches:
            #     #     first_match_index = matches.index(True)
            #     #     name = known_face_names[first_match_index]

            #     # Or instead, use the known face with the smallest distance to the new face
            #     faceDistances = face_recognition.face_distance(knownFaces, faceEncodings)
            #     bestMatchIndex = np.argmin(faceDistances)
            #     if matches[bestMatchIndex]:
            #         name = knownFaces[bestMatchIndex]

            #     faceNames.append(name)

            # for (top, right, bottom, left), name in zip(faceLocations, faceNames):
            #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            #     font = cv2.FONT_HERSHEY_DUPLEX
            #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Encode the frame in JPEG format
            frame = detectAndRecognizeFace(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield the frame in byte format with the appropriate header for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # Video streaming route
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/test')
def test():
    return "test response"


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
