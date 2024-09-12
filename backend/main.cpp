#include <opencv2/opencv.hpp>
#include <vector>

extern "C" {
    int capture_frame(unsigned char** data, int* size) {
        cv::VideoCapture cap(1);

        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open the camera." << std::endl;
            return -1;
        }

        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            return -1;
        }

        // Encode the frame as a JPEG
        std::vector<unsigned char> buf;
        cv::imencode(".jpg", frame, buf);

        // Allocate memory for the image and pass the data back
        *size = buf.size();
        *data = (unsigned char*)malloc(*size);
        std::copy(buf.begin(), buf.end(), *data);

        cap.release();
        return 0;  // Success
    }
}
