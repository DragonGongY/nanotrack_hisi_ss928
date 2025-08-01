#include "backbone.h"

#include <iostream>

Backbone::Backbone(const char* modelPath) {
  aclError ret = aclmdlQuerySize(modelPath, &modelWorkSize_, &modelWeightSize_);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("query model failed, model file is %s, errorCode is %d",
              modelPath, static_cast<int32_t>(ret));
  }
  // using ACL_MEM_MALLOC_HUGE_FIRST to malloc memory, huge memory is preferred
  // to use and huge memory can improve performance.
  ret = aclrtMalloc(&modelWorkPtr_, modelWorkSize_, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG(
        "malloc buffer for work failed, require size is %zu, errorCode is %d",
        modelWorkSize_, static_cast<int32_t>(ret));
  }

  // using ACL_MEM_MALLOC_HUGE_FIRST to malloc memory, huge memory is preferred
  // to use and huge memory can improve performance.
  ret = aclrtMalloc(&modelWeightPtr_, modelWeightSize_,
                    ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG(
        "malloc buffer for weight failed, require size is %zu, errorCode is %d",
        modelWeightSize_, static_cast<int32_t>(ret));
  }

  ret = aclmdlLoadFromFileWithMem(modelPath, &modelId_, modelWorkPtr_,
                                  modelWorkSize_, modelWeightPtr_,
                                  modelWeightSize_);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("load model from file failed, model file is %s, errorCode is %d",
              modelPath, static_cast<int32_t>(ret));
  }

  modelDesc_ = aclmdlCreateDesc();
  if (modelDesc_ == nullptr) {
    ERROR_LOG("create model description failed");
  }

  ret = aclmdlGetDesc(modelDesc_, modelId_);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("get model description failed, modelId is %u, errorCode is %d",
              modelId_, static_cast<int32_t>(ret));
  }
}
Backbone::~Backbone() {
  aclError ret;
  // release resource includes acl resource, data set and unload model
  aclrtFree(inputBuffer_b);
  inputBuffer_b = nullptr;
  (void)aclmdlDestroyDataset(inputDataset_b);
  inputDataset_b = nullptr;

  aclrtFree(outputBuffer_b);
  outputBuffer_b = nullptr;
  (void)aclmdlDestroyDataset(outputDataset_b);
  outputDataset_b = nullptr;

  ret = aclmdlDestroyDesc(modelDesc_);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("destroy description failed, errorCode is %d", ret);
  }

  ret = aclmdlUnload(modelId_);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("unload model failed, errorCode is %d", ret);
  }
}

Result Backbone::backbone_initDatasets() {
  INFO_LOG("START backbone_initDatasets ");
  aclError ret;
  // create data set of input
  inputDataset_b = aclmdlCreateDataset();
  inputBufferSize_b = aclmdlGetInputSizeByIndex(modelDesc_, 0);
  aclrtMalloc(&inputBuffer_b, inputBufferSize_b, ACL_MEM_MALLOC_HUGE_FIRST);
  aclDataBuffer* inputData =
      aclCreateDataBuffer(inputBuffer_b, inputBufferSize_b);
  ret = aclmdlAddDatasetBuffer(inputDataset_b, inputData);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("backbone_initDatasets inputDataset_b failed, errorCode is %d",
              ret);
    return FAILED;
  } else {
    INFO_LOG("backbone_initDatasets inputDataset_b success");
  }

  // create data set of output
  outputDataset_b = aclmdlCreateDataset();
  modelOutputSize_b = aclmdlGetOutputSizeByIndex(modelDesc_, 0);
  aclrtMalloc(&outputBuffer_b, modelOutputSize_b, ACL_MEM_MALLOC_HUGE_FIRST);
  aclDataBuffer* outputData =
      aclCreateDataBuffer(outputBuffer_b, modelOutputSize_b);
  ret = aclmdlAddDatasetBuffer(outputDataset_b, outputData);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("backbone_initDatasets outputDataset_b failed, errorCode is %d",
              ret);
    return FAILED;
  } else {
    INFO_LOG("backbone_initDatasets outputDataset_b success");
  }
  INFO_LOG("FINISH backbone_initDatasets ");
  return SUCCESS;
}

Result Backbone::backbone_ProcessInput(cv::Mat& img) {
  INFO_LOG("START Preprocess the input img ");

  // get properties of image
  int32_t channel = img.channels();
  int32_t Height = img.rows;
  int32_t Weight = img.cols;
  imageBytes = (float*)malloc(1 * channel * Height * Weight * sizeof(float));
  memset(imageBytes, 0, 1 * channel * Height * Weight * sizeof(float));

  // 图像转换为字节，从 HWC 到 NCHW
  for (int h = 0; h < Height; ++h) {
    for (int w = 0; w < Weight; ++w) {
      for (int c = 0; c < channel; ++c) {
        // 将像素值从 cv::Vec3b (即 uint8_t) 转换为 float
        imageBytes[c * Height * Weight + h * Weight + w] =
            static_cast<float>(img.at<cv::Vec3b>(h, w)[c]);
      }
    }
  }
  INFO_LOG("FINISH Preprocess the input img ");
  return SUCCESS;
}

Result Backbone::backbone_Inference() {
  INFO_LOG("START ACNNModel_B::backbone_Inference");
  // copy host datainputs to device
  aclError ret = aclrtMemcpy(inputBuffer_b, inputBufferSize_b, this->imageBytes,
                             inputBufferSize_b, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("memcpy  failed, errorCode is %d", ret);
    return FAILED;
  }
  // inference
  ret = aclmdlExecute(modelId_, inputDataset_b, outputDataset_b);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("execute model failed, errorCode is %d", ret);
    return FAILED;
  }
  INFO_LOG("FINISH ACNNModel_B::backbone_Inference");

  return SUCCESS;
}

void* Backbone::backbone_GetResults() {
  aclError ret;
  void* outHostData = nullptr;
  float* outData = nullptr;
  aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(outputDataset_b, 0);
  void* data = aclGetDataBufferAddr(dataBuffer);
  uint32_t output_length = aclGetDataBufferSizeV2(dataBuffer);

  // copy device output data to host
  aclrtMallocHost(&outHostData, output_length);
  ret = aclrtMemcpy(outHostData, output_length, data, output_length,
                    ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("memcpy  failed, errorCode is %d", ret);
    return nullptr;
  }

  INFO_LOG("FINISH ACNNModel_B::backbone_GetResults");
  return outHostData;
}

void* Backbone::runBackbone(cv::Mat& img) {
  //根据backbone与head的输入区别进行前处理
  Result ret;
  //前处理
  ret = backbone_ProcessInput(img);
  if (ret != SUCCESS) {
    ERROR_LOG("ProcessInput  failed");
    return nullptr;
  }
  //推理
  ret = backbone_Inference();
  if (ret != SUCCESS) {
    ERROR_LOG("Inference  failed");
    return nullptr;
  }
  //后处理
  float* outData = backbone_GetResults();
  if (ret != SUCCESS) {
    ERROR_LOG("Inference  failed");
    return nullptr;
  }
  return outData;
}
