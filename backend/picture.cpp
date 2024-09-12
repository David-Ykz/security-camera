#include <opencv2/opencv.hpp>
#include <iostream>


int main() {
    // Open the camera device (you may need to adjust the index)
    cv::VideoCapture cap(1);  // or /dev/video1 or /dev/video0

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera." << std::endl;
        return -1;
    }

    // Capture frames in a loop
    cv::Mat frame;
    while (true) {
        cap >> frame;  // Capture a frame
        if (frame.empty()) {
            std::cerr << "Error: Captured an empty frame." << std::endl;
            break;
        }

        // You can save the frame as an image file or process it
        cv::imwrite("frame.jpg", frame);

        // Break after one capture for testing purposes (remove in real use case)
        break;
    }

    cap.release();  // Release the camera
    return 0;
}
