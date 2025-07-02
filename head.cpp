#include "head.h"

#include <opencv2/opencv.hpp>

Head::Head(const char* modelPath) {
    aclError ret = aclmdlQuerySize(modelPath, &modelWorkSize_, &modelWeightSize_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("query model failed, model file is %s, errorCode is %d", modelPath,
                  static_cast<int32_t>(ret));
    }
    // using ACL_MEM_MALLOC_HUGE_FIRST to malloc memory, huge memory is preferred to use
    // and huge memory can improve performance.
    ret = aclrtMalloc(&modelWorkPtr_, modelWorkSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc buffer for work failed, require size is %zu, errorCode is %d",
                  modelWorkSize_, static_cast<int32_t>(ret));
    }

    // using ACL_MEM_MALLOC_HUGE_FIRST to malloc memory, huge memory is preferred to use
    // and huge memory can improve performance.
    ret = aclrtMalloc(&modelWeightPtr_, modelWeightSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc buffer for weight failed, require size is %zu, errorCode is %d",
                  modelWeightSize_, static_cast<int32_t>(ret));
    }

    ret = aclmdlLoadFromFileWithMem(modelPath, &modelId_, modelWorkPtr_, modelWorkSize_,
                                    modelWeightPtr_, modelWeightSize_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("load model from file failed, model file is %s, errorCode is %d", modelPath,
                  static_cast<int32_t>(ret));
    }

    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
    }

    ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("get model description failed, modelId is %u, errorCode is %d", modelId_,
                  static_cast<int32_t>(ret));
    }
}
Head::~Head() {
    aclError ret;
    // release resource includes acl resource, data set and unload model
    aclrtFree(inputBuffer_n1);
    inputBuffer_n1 = nullptr;
    aclrtFree(inputBuffer_n2);
    inputBuffer_n2 = nullptr;
    (void)aclmdlDestroyDataset(inputDataset_n);
    inputDataset_n = nullptr;

    aclrtFree(outputBuffer_n1);
    outputBuffer_n1 = nullptr;
    aclrtFree(outputBuffer_n2);
    outputBuffer_n2 = nullptr;
    (void)aclmdlDestroyDataset(outputDataset_n);
    outputDataset_n = nullptr;

    ret = aclmdlDestroyDesc(modelDesc_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("destroy description failed, errorCode is %d", ret);
    }

    ret = aclmdlUnload(modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("unload model failed, errorCode is %d", ret);
    }
}

Result Head::head_initDatasets() {
    aclError ret;
    // create data set of input
    inputDataset_n = aclmdlCreateDataset();

    inputBufferSize_n1 = aclmdlGetInputSizeByIndex(modelDesc_, 0);
    const char* inputname_1 = aclmdlGetInputNameByIndex(modelDesc_, 0);
    aclrtMalloc(&inputBuffer_n1, inputBufferSize_n1, ACL_MEM_MALLOC_HUGE_FIRST);
    aclDataBuffer* inputData_n1 = aclCreateDataBuffer(inputBuffer_n1, inputBufferSize_n1);
    ret = aclmdlAddDatasetBuffer(inputDataset_n, inputData_n1);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("head_initDatasets aclmdlAddDatasetBuffer n1 failed, errorCode is %d", ret);
        return FAILED;
    } else {
        INFO_LOG("head_initDatasets aclmdlAddDatasetBuffer n1 success");
    }

    inputBufferSize_n2 = aclmdlGetInputSizeByIndex(modelDesc_, 1);
    const char* inputname_2 = aclmdlGetInputNameByIndex(modelDesc_, 1);
    aclrtMalloc(&inputBuffer_n2, inputBufferSize_n2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclDataBuffer* inputData_n2 = aclCreateDataBuffer(inputBuffer_n2, inputBufferSize_n2);
    ret = aclmdlAddDatasetBuffer(inputDataset_n, inputData_n2);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("head_initDatasets aclmdlAddDatasetBuffer n2 failed, errorCode is %d", ret);
        return FAILED;
    } else {
        INFO_LOG("head_initDatasets aclmdlAddDatasetBuffer n2 success");
    }

    // create data set of output
    outputDataset_n = aclmdlCreateDataset();
    modelOutputSize_n1 = aclmdlGetOutputSizeByIndex(modelDesc_, 0);
    aclrtMalloc(&outputBuffer_n1, modelOutputSize_n1, ACL_MEM_MALLOC_HUGE_FIRST);
    aclDataBuffer* outputData_n1 = aclCreateDataBuffer(outputBuffer_n1, modelOutputSize_n1);
    ret = aclmdlAddDatasetBuffer(outputDataset_n, outputData_n1);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlAddDatasetBuffer n1 failed, errorCode is %d", ret);
        return FAILED;
    } else {
        INFO_LOG("aclmdlAddDatasetBuffer n1  success");
    }

    modelOutputSize_n2 = aclmdlGetOutputSizeByIndex(modelDesc_, 1);
    aclrtMalloc(&outputBuffer_n2, modelOutputSize_n2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclDataBuffer* outputData_n2 = aclCreateDataBuffer(outputBuffer_n2, modelOutputSize_n2);
    ret = aclmdlAddDatasetBuffer(outputDataset_n, outputData_n2);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlAddDatasetBuffer n2 failed, errorCode is %d", ret);
        return FAILED;
    } else {
        INFO_LOG("aclmdlAddDatasetBuffer n2 success");
    }

    return SUCCESS;
}

Result Head::head_Inference(float* input_data0, float* input_data1) {
    // copy host datainputs to device
    aclError ret = aclrtMemcpy(inputBuffer_n1, inputBufferSize_n1, input_data0, inputBufferSize_n1,
                               ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("memcpy  failed, errorCode is %d", ret);
        return FAILED;
    }
    ret = aclrtMemcpy(inputBuffer_n2, inputBufferSize_n2, input_data1, inputBufferSize_n2,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("memcpy  failed, errorCode is %d", ret);
        return FAILED;
    }
    // inference
    ret = aclmdlExecute(modelId_, inputDataset_n, outputDataset_n);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("execute model failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result Head::head_GetResults(std::vector<cv::Mat>& output) {
    aclError ret;
    void* outHostData = nullptr;
    float* outData = nullptr;
    uint32_t output_num = aclmdlGetNumOutputs(modelDesc_);
    output.resize(output_num);  // 假设第一个维度是 batch size
    for (int i = 0; i < output_num; ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(outputDataset_n, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        uint32_t output_length = aclGetDataBufferSizeV2(dataBuffer);

        //查询模型输出数据的维度与数据类型
        aclmdlIODims dims;
        ret = aclmdlGetOutputDims(modelDesc_, i, &dims);  // 创建一个aclmdlIODims实例
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("aclmdlGetOutputDims failed, errorCode is %d", ret);
        }
        std::vector<int> dimensions(4);  // 创建一个大小为 4 的 vector
        for (size_t i = 0; i < dims.dimCount; ++i) {
            dimensions[i] = static_cast<int>(dims.dims[i]);  // 将int64_t转换为int并存储到vector中
        }

        aclDataType datatype = aclmdlGetOutputDataType(modelDesc_, i);

        // copy device output data to host
        aclrtMallocHost(&outHostData, output_length);
        ret =
            aclrtMemcpy(outHostData, output_length, data, output_length, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("memcpy  failed, errorCode is %d", ret);
            return FAILED;
        }
        outData = reinterpret_cast<float*>(outHostData);

        //+++++++++outputs size0: 1 x 2 x 16 x 16
        //+++++++++outputs size1: 1 x 4 x 16 x 16
        if (i == 0) {
            int sizes[4] = {1, 2, 16, 16};
            cv::Mat mat(4, sizes, CV_32F, outData);
            output[i] = mat.reshape(1, {1, 2, 16, 16});
        } else if (i == 1) {
            int sizes[4] = {1, 4, 16, 16};
            cv::Mat mat(4, sizes, CV_32F, outData);
            output[i] = mat.reshape(1, {1, 4, 16, 16});
        }
    }
    return SUCCESS;
}

Result Head::runHead(std::vector<cv::Mat>& output, float*& input_data0, float*& input_data1) {
    Result ret;

    //推理
    ret = head_Inference(input_data0, input_data1);
    if (ret != SUCCESS) {
        ERROR_LOG("Inference  failed");
        return FAILED;
    }
    ret = head_GetResults(output);
    if (ret != SUCCESS) {
        ERROR_LOG("Inference  failed");
        return FAILED;
    }
    return SUCCESS;
}
