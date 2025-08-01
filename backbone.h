#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "acl.h"

#define INFO_LOG(fmt, ...)                             \
  fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__); \
  fflush(stdout)
#define ERROR_LOG(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)

typedef enum Result { SUCCESS = 0, FAILED = 1 } Result;

class Backbone {
 public:
  Backbone(const char* modelPath);
  ~Backbone();
  Result backbone_initDatasets();
  Result backbone_ProcessInput(cv::Mat& img);
  Result backbone_Inference();
  Result backbone_GetResults(std::vector<std::vector<float>>& output);
  void* backbone_GetResults();
  void* runBackbone(cv::Mat& img);

 private:
  uint32_t modelId_;
  size_t modelWorkSize_;    // model work memory buffer size
  size_t modelWeightSize_;  // model weight memory buffer size
  void* modelWorkPtr_;      // model work memory buffer
  void* modelWeightPtr_;    // model weight memory buffer
  aclmdlDesc* modelDesc_;

  aclmdlDataset* inputDataset_b;
  aclmdlDataset* outputDataset_b;
  void* inputBuffer_b;
  void* outputBuffer_b;
  size_t inputBufferSize_b;
  size_t modelOutputSize_b;
  float* imageBytes;
};