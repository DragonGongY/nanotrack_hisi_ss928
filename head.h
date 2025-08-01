#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "acl.h"
#include "backbone.h"

#define INFO_LOG(fmt, ...)                             \
  fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__); \
  fflush(stdout)
#define ERROR_LOG(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)

class Head {
 public:
  Head(const char* modelPath);
  ~Head();
  Result head_initDatasets();
  Result head_Inference(void* input_data0, void* input_data1);
  Result head_GetResults(std::vector<cv::Mat>& output);
  Result runHead(std::vector<cv::Mat>& output, void*& input_data0,
                 void*& input_data1);

 private:
  uint32_t modelId_;
  size_t modelWorkSize_;    // model work memory buffer size
  size_t modelWeightSize_;  // model weight memory buffer size
  void* modelWorkPtr_;      // model work memory buffer
  void* modelWeightPtr_;    // model weight memory buffer
  aclmdlDesc* modelDesc_;

  aclmdlDataset* inputDataset_n;
  aclmdlDataset* outputDataset_n;
  void* inputBuffer_n1;
  void* outputBuffer_n1;
  size_t inputBufferSize_n1;
  size_t modelOutputSize_n1;
  void* inputBuffer_n2;
  void* outputBuffer_n2;
  size_t inputBufferSize_n2;
  size_t modelOutputSize_n2;
};