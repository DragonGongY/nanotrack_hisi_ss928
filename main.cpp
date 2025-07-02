#include <iostream>
#include "nanotrack_app.h"

int main(int argc, char* argv[]) {
    NanoTrackApp nanotrack_app;
    nanotrack_app.initialize();
    cv::Mat first_frame = cv::imread("/app/sd/imgs/0.jpg");
    nanotrack_app.init(first_frame, cv::Rect(499, 96, 137, 226));
    for (int i = 1; i < 21; i++) {
        cv::Rect track_bbox;
        float track_score;
        cv::Mat frame = cv::imread("/app/sd/imgs/" + std::to_string(i) + ".jpg");
        nanotrack_app.track(frame, track_bbox, track_score);
        cv::rectangle(frame, track_bbox, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, std::to_string(track_score), cv::Point(track_bbox.x, track_bbox.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        std::cout << "++++++++++++++++ score: " << track_score << std::endl;
        cv::imwrite("/app/sd/results/" + std::to_string(i) + ".jpg", frame);
    }
    return 0;
}