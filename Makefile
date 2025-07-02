SOURCE_TREE ?= $(CURDIR)/../../../..
include $(SOURCE_TREE)/cmake/build/base.mak
CURR_ROOT := $(shell pwd)

PDT_MODULE_PATH := $(SOURCE_TREE)/source/camera/demo/drone/modules
INC_CFLAGS += -I$(CURR_ROOT)
INC_CFLAGS += -I$(OPENSOURCE_ROOT)/eigen/eigen-3.3.9
INC_CFLAGS += -I$(SOURCE_TREE)/source/thirdparty/slink
INC_CFLAGS += -I$(SOURCE_TREE)/source/vision/component/tracking/trinidy/common/av200
INC_CFLAGS += -I$(OPENSOURCE_ROOT)/opencv/out/board/include/opencv4

SO_LIB += -L$(OPENSOURCE_ROOT)/opencv/out/board/lib
SO_LIB += -lopencv_core -lopencv_features2d -lopencv_imgcodecs -lopencv_imgproc -lopencv_calib3d -lopencv_highgui \
			-lopencv_flann -lopencv_photo -lopencv_stitching  -lopencv_video

SO_LIB += -L$(MPP_OUT)/lib/npu -lacl_cblas -lascend_protobuf -lge_executor \
          -lacl_retr -lcce_aicore -lgraph -lacl_tdt_queue -lcpu_kernels_context -lmmpa \
          -ladump -lcpu_kernels -lmsprofiler \
          -laicpu_kernels -lc_sec -lmsprof \
          -laicpu_processer -ldrv_aicpu -lopt_feature \
          -laicpu_prof -ldrvdevdrv -lregister \
          -laicpu_scheduler -ldrv_dfx -lruntime \
          -lalog -lerror_manager -lslog \
          -lascendcl -lge_common -ltsdclient 

SRC_ROOT 	:= $(CURR_ROOT)
SRC_DIR     := $(SRC_ROOT)
ANN_DIR     := $(SOURCE_TREE)/source/vision/component/tracking/trinidy/common/av200
# SRCS := $(shell find $(SRC_DIR) -name '*.cpp') $(shell find $(ANN_DIR) -name '*.cpp')
SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
TARGET := yolov5
include $(PWD)/../build/base_cpp.mak


