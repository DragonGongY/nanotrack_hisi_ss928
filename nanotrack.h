#pragma once

#include <dirent.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "acl.h"
#include "backbone.h"
#include "head.h"

class NanoTrack {
 public:
  NanoTrack(const char* Tback_model, const char* Xback_model,
            const char* Head_model);
  ~NanoTrack();
  void initsource();

  void init(const cv::Mat& img, const cv::Rect2f& bbox);
  void track(const cv::Mat& img, cv::Rect& track_bbox, float& track_score);

  const char* g_modelPath_1;
  const char* g_modelPath_2;
  const char* g_modelPath_3;

 private:
  Backbone module_T127;
  Backbone module_X255;
  Head module_head;
  void* result_T;
  void* result_X;

  cv::Point2f center_pos;
  cv::Size2f size;
  cv::Scalar channel_average;
  int score_size, cls_out_channels;
  std::vector<float> window;
  cv::Mat points;

  std::vector<float> createHanningWindow();
  cv::Mat generate_points(int stride, int size);
  cv::Mat get_subwindow(const cv::Mat& im, cv::Point2f pos, int model_sz,
                        int original_sz, cv::Scalar avg_chans);
  std::vector<float> convert_score(const cv::Mat& score);
  cv::Mat convert_bbox(const cv::Mat& delta, const cv::Mat& point);
  std::tuple<float, float, float, float> bbox_clip(float cx, float cy,
                                                   float width, float height,
                                                   cv::Size boundary);
};
