import ctypes
import cv2
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

# Load the shared library (compiled C++ code)
video_capture = ctypes.CDLL('./video_capture.so')

# Define the C++ function signature (return types)
video_capture.capture_frame.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_int)]
video_capture.capture_frame.restype = ctypes.c_int

def capture_frame():
    # Initialize pointers for the byte array and size
    data_ptr = ctypes.POINTER(ctypes.c_ubyte)()
    size = ctypes.c_int()

    # Call the C++ function
    if video_capture.capture_frame(ctypes.byref(data_ptr), ctypes.byref(size)) == 0:
        # Convert the byte array back to a numpy array (which OpenCV uses)
        byte_array = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_ubyte * size.value)).contents
        np_array = np.frombuffer(byte_array, dtype=np.uint8)

        # Decode the JPEG image to a format Flask can stream (OpenCV Mat format)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Free the memory allocated by C++ (if applicable)
        ctypes.free(data_ptr)

        # Encode frame back to JPEG format for streaming
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
    else:
        return None

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = capture_frame()
            if frame:
                # Streaming MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
