#include "nanotrack_app.h"

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

NanoTrackApp::NanoTrackApp()
    : deviceId_(0),
      context_(nullptr),
      stream_(nullptr),
      nanotrack_(nullptr) {
    T_model_path_ = "/app/sd/nanotrack_fp32/backT.om";
    X_model_path_ = "/app/sd/nanotrack_fp32/backX.om";
    head_model_path_ = "/app/sd/nanotrack_fp32/head.om";
}

NanoTrackApp::~NanoTrackApp() {
    deinitialize(); 
}

bool NanoTrackApp::fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

Result NanoTrackApp::initialize() {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) return FAILED;

    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_SUCCESS) return FAILED;

    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_SUCCESS) return FAILED;

    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS) return FAILED;

    if (!fileExists(T_model_path_) || !fileExists(X_model_path_) || !fileExists(head_model_path_)) {
        std::cerr << "One or more model files not found.\n";
        return FAILED;
    }

    nanotrack_ = new NanoTrack(T_model_path_, X_model_path_, head_model_path_);
    nanotrack_->initsource();

    return SUCCESS;
}

void NanoTrackApp::init(const cv::Mat& frame, cv::Rect init_bbox) {
    nanotrack_->init(frame, init_bbox);
}

Result NanoTrackApp::track(const cv::Mat& frame, cv::Rect &track_bbox, float &track_score) {
    double t1 = cv::getTickCount();
    nanotrack_->track(frame, track_bbox, track_score);
    double t2 = cv::getTickCount();
    double ms = (t2 - t1) * 1000 / cv::getTickFrequency();
    std::cout << "Frame processed in " << ms << " ms\n";

    std::cout << "++++++++++++++++track_bbox: " << track_bbox << std::endl;
    std::cout << "++++++++++++++++track_score: " << track_score << std::endl;
    return SUCCESS;
}

Result NanoTrackApp::deinitialize() {
    delete nanotrack_;
    nanotrack_ = nullptr;

    if (stream_ != nullptr) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
    if (context_ != nullptr) {
        aclrtDestroyContext(context_);
        context_ = nullptr;
    }
    aclrtResetDevice(deviceId_);
    aclFinalize();

    return SUCCESS;
}
