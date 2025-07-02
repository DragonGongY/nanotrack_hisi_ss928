#pragma once

#include <string>
#include "acl.h"
#include "nanotrack.h"

class NanoTrackApp {
public:
    NanoTrackApp();
    ~NanoTrackApp();

    Result initialize();
    void init(const cv::Mat& frame, cv::Rect init_bbox);
    Result track(const cv::Mat& frame, cv::Rect &track_bbox, float &track_score);
    Result deinitialize();

private:
    bool fileExists(const std::string& path);

    const char* T_model_path_;
    const char* X_model_path_;
    const char* head_model_path_;

    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;

    NanoTrack* nanotrack_;
};
