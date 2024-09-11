#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Open the default camera (0 means the first connected camera)
    cv::VideoCapture cap(1);

    // Check if camera opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        // Capture frame-by-frame
        cap >> frame;

        // If the frame is empty, break the loop
        if (frame.empty())
            break;

        // Display the resulting frame
//        cv::imshow("Camera Feed", frame);

        // Exit the loop if the user presses the 'q' key
        if (cv::waitKey(30) == 'q') {
            break;
        }
    }

    // Release the camera
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
