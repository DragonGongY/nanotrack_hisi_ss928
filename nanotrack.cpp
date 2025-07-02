#include "nanotrack.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

// ----------- config参数 -----------
const int INSTANCE_SIZE = 255;
const int EXEMPLAR_SIZE = 127;
const int BASE_SIZE = 7;
const int POINT_STRIDE = 16;
const float CONTEXT_AMOUNT = 0.5f;
const float PENALTY_K = 0.15f;
const float WINDOW_INFLUENCE = 0.455f;
const float LR = 0.37f;

std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> corner2center(const cv::Mat& delta) {
    cv::Mat cx = (delta.row(0) + delta.row(2)) / 2;
    cv::Mat cy = (delta.row(1) + delta.row(3)) / 2;
    cv::Mat w = delta.row(2) - delta.row(0);
    cv::Mat h = delta.row(3) - delta.row(1);
    return std::make_tuple(cx, cy, w, h);
}

NanoTrack::NanoTrack::NanoTrack(const char* modelPath_1, const char* modelPath_2,
                                const char* modelPath_3)
    : g_modelPath_1(modelPath_1),
      g_modelPath_2(modelPath_2),
      g_modelPath_3(modelPath_3),
      module_T127(modelPath_1),
      module_X255(modelPath_2),
      module_head(modelPath_3) {
    score_size = (INSTANCE_SIZE - EXEMPLAR_SIZE) / POINT_STRIDE + 1 + BASE_SIZE;
    window = createHanningWindow();
    cls_out_channels = 2;
    points = generate_points(POINT_STRIDE, score_size);
}

NanoTrack::~NanoTrack() {}

void NanoTrack::initsource() {
    Result ret;

    ret = module_T127.backbone_initDatasets();
    if (ret != SUCCESS) {
        ERROR_LOG("module_T127.backbone_initDatasets failed ");
    }
    ret = module_X255.backbone_initDatasets();
    if (ret != SUCCESS) {
        ERROR_LOG("module_X255.backbone_initDatasets failed ");
    }
    ret = module_head.head_initDatasets();
    if (ret != SUCCESS) {
        ERROR_LOG("module_head.head_initDatasets failed ");
    }
}

void NanoTrack::init(const cv::Mat& img, const cv::Rect2f& bbox) {
    center_pos = cv::Point2f(bbox.x + (bbox.width - 1) / 2.0f, bbox.y + (bbox.height - 1) / 2.0f);
    size = cv::Size2f(bbox.width, bbox.height);

    float w_z = size.width + CONTEXT_AMOUNT * (size.width + size.height);
    float h_z = size.height + CONTEXT_AMOUNT * (size.width + size.height);
    int s_z = round(std::sqrt(w_z * h_z));

    channel_average = mean(img);

    cv::Mat z_crop = get_subwindow(img, center_pos, EXEMPLAR_SIZE, s_z, channel_average);

    result_T = module_T127.runBackbone(z_crop);
}

void NanoTrack::track(const cv::Mat& img, cv::Rect &track_bbox, float &track_score) {
    float w_z = size.width + CONTEXT_AMOUNT * (size.width + size.height);
    float h_z = size.height + CONTEXT_AMOUNT * (size.width + size.height);
    float s_z = std::sqrt(w_z * h_z);
    float scale_z = EXEMPLAR_SIZE / s_z;
    float s_x = s_z * (INSTANCE_SIZE / (float)EXEMPLAR_SIZE);

    cv::Mat x_crop = get_subwindow(img, center_pos, INSTANCE_SIZE, round(s_x), channel_average);
    result_X = module_X255.runBackbone(x_crop);

    std::vector<cv::Mat> outputs;
    module_head.runHead(outputs, result_T, result_X);
    std::vector<float> score = convert_score(outputs[0]);
    cv::Mat pred_bbox = convert_bbox(outputs[1], points);

    auto change = [](float r) { return std::max(r, 1.0f / r); };
    auto sz = [](float w, float h) {
        float pad = (w + h) * 0.5f;
        return std::sqrt((w + pad) * (h + pad));
    };

    std::vector<float> s_c, r_c, penalty, pscore;
    for (int i = 0; i < pred_bbox.cols; ++i) {
        float sc = change(sz(pred_bbox.at<float>(2, i), pred_bbox.at<float>(3, i)) /
                          sz(size.width * scale_z, size.height * scale_z));
        float rc = change((size.width / size.height) /
                          (pred_bbox.at<float>(2, i) / pred_bbox.at<float>(3, i)));
        s_c.push_back(sc);
        r_c.push_back(rc);
        penalty.push_back(std::exp(-(rc * sc - 1) * PENALTY_K));
        pscore.push_back(penalty.back() * score[i]);
    }
    for (size_t i = 0; i < pscore.size(); ++i)
        pscore[i] = pscore[i] * (1 - WINDOW_INFLUENCE) + window[i] * WINDOW_INFLUENCE;

    int best_idx = std::max_element(pscore.begin(), pscore.end()) - pscore.begin();
    std::vector<float> bbox(4);
    for (int i = 0; i < 4; ++i) bbox[i] = pred_bbox.at<float>(i, best_idx) / scale_z;

    float lr = penalty[best_idx] * score[best_idx] * LR;
    float cx = bbox[0] + center_pos.x;
    float cy = bbox[1] + center_pos.y;
    float width = size.width * (1 - lr) + bbox[2] * lr;
    float height = size.height * (1 - lr) + bbox[3] * lr;

    std::tie(cx, cy, width, height) = bbox_clip(cx, cy, width, height, img.size());

    center_pos = cv::Point2f(cx, cy);
    size = cv::Size2f(width, height);

    std::vector<float> out_bbox = {cx - width / 2, cy - height / 2, width, height};
    float best_score = score[best_idx];

    track_bbox = cv::Rect(out_bbox[0], out_bbox[1], out_bbox[2], out_bbox[3]);
    track_score = best_score;
}

std::vector<float> NanoTrack::createHanningWindow() {
    cv::Mat hanning;
    cv::createHanningWindow(hanning, cv::Size(score_size, score_size), CV_32F);
    return std::vector<float>((float*)hanning.datastart, (float*)hanning.dataend);
}

cv::Mat NanoTrack::generate_points(int stride, int size) {
    cv::Mat points(size * size, 2, CV_32F);
    int idx = 0;
    int ori = -(size / 2) * stride;
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x, ++idx) {
            points.at<float>(idx, 0) = ori + stride * x;
            points.at<float>(idx, 1) = ori + stride * y;
        }
    return points;
}

cv::Mat NanoTrack::get_subwindow(const cv::Mat& im, cv::Point2f pos, int model_sz, int original_sz,
                                 cv::Scalar avg_chans) {
    int im_h = im.rows, im_w = im.cols;
    float c = (original_sz + 1) / 2.0f;
    float context_xmin = std::floor(pos.x - c + 0.5f);
    float context_ymin = std::floor(pos.y - c + 0.5f);
    float context_xmax = context_xmin + original_sz - 1;
    float context_ymax = context_ymin + original_sz - 1;

    int left_pad = int(std::max(0.f, -context_xmin));
    int top_pad = int(std::max(0.f, -context_ymin));
    int right_pad = int(std::max(0.f, context_xmax - im_w + 1));
    int bottom_pad = int(std::max(0.f, context_ymax - im_h + 1));

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;

    cv::Mat te_im;
    if (top_pad > 0 || bottom_pad > 0 || left_pad > 0 || right_pad > 0) {
        te_im = cv::Mat::zeros(im_h + top_pad + bottom_pad, im_w + left_pad + right_pad, im.type());
        im.copyTo(te_im(cv::Rect(left_pad, top_pad, im_w, im_h)));
        if (top_pad > 0) te_im(cv::Rect(left_pad, 0, im_w, top_pad)).setTo(avg_chans);
        if (bottom_pad > 0)
            te_im(cv::Rect(left_pad, im_h + top_pad, im_w, bottom_pad)).setTo(avg_chans);
        if (left_pad > 0) te_im(cv::Rect(0, 0, left_pad, te_im.rows)).setTo(avg_chans);
        if (right_pad > 0)
            te_im(cv::Rect(im_w + left_pad, 0, right_pad, te_im.rows)).setTo(avg_chans);
    } else {
        te_im = im;
    }
    cv::Mat im_patch =
        te_im(cv::Rect(int(context_xmin), int(context_ymin), int(context_xmax - context_xmin + 1),
                       int(context_ymax - context_ymin + 1)));

    if (model_sz != original_sz) cv::resize(im_patch, im_patch, cv::Size(model_sz, model_sz));

    return im_patch;
}

std::vector<float> NanoTrack::convert_score(const cv::Mat& score) {
    cv::Mat s = score.reshape(1, {cls_out_channels, score.size[2] * score.size[3]});
    cv::Mat s_t;
    cv::transpose(s, s_t);  // (N, C)
    std::vector<float> out;
    for (int i = 0; i < s_t.rows; ++i) {
        float maxv = *std::max_element(s_t.ptr<float>(i), s_t.ptr<float>(i) + cls_out_channels);
        float sum = 0.0f;
        std::vector<float> exps(cls_out_channels);
        for (int j = 0; j < cls_out_channels; ++j) {
            exps[j] = std::exp(s_t.at<float>(i, j) - maxv);
            sum += exps[j];
        }
        out.push_back(exps[1] / sum);  // 取正类概率
    }
    return out;
}

cv::Mat NanoTrack::convert_bbox(const cv::Mat& delta, const cv::Mat& point) {
    cv::Mat d = delta.reshape(1, {4, delta.size[2] * delta.size[3]});
    cv::Mat d_out = d.clone();
    for (int i = 0; i < d.cols; ++i) {
        d_out.at<float>(0, i) = point.at<float>(i, 0) - d.at<float>(0, i);
        d_out.at<float>(1, i) = point.at<float>(i, 1) - d.at<float>(1, i);
        d_out.at<float>(2, i) = point.at<float>(i, 0) + d.at<float>(2, i);
        d_out.at<float>(3, i) = point.at<float>(i, 1) + d.at<float>(3, i);
    }
    cv::Mat cx, cy, w, h;
    std::tie(cx, cy, w, h) = corner2center(d_out);
    cv::Mat out;
    cv::vconcat(std::vector<cv::Mat>{cx, cy, w, h}, out);  // 4 x N
    return out;
}

std::tuple<float, float, float, float> NanoTrack::bbox_clip(float cx, float cy, float width,
                                                            float height, cv::Size boundary) {
    cx = std::max(0.f, std::min(cx, (float)boundary.width));
    cy = std::max(0.f, std::min(cy, (float)boundary.height));
    width = std::max(10.f, std::min(width, (float)boundary.width));
    height = std::max(10.f, std::min(height, (float)boundary.height));
    return std::make_tuple(cx, cy, width, height);
}