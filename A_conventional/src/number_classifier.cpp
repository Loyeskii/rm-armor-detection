#include "number_classifier.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>

namespace rm {

DigitClassifier::DigitClassifier(const std::string& onnx_path,
                                 const std::vector<std::string>& labels,
                                 float min_conf)
    : labels_(labels), min_conf_(min_conf) {
    try {
        net_ = cv::dnn::readNetFromONNX(onnx_path);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "[info] 数字网络已加载: " << onnx_path << "\n";
    } catch (const cv::Exception& e) {
        std::cerr << "[warn] 数字网络加载失败: " << onnx_path << "\n"
                  << "       " << e.what() << "\n";
        net_ = cv::dnn::Net();
    }
}

bool DigitClassifier::classify(const cv::Mat& bgr, Armor& armor) const {
    if (net_.empty()) return false;

    // ============================================================
    // 预处理必须与训练时完全一致
    // ------------------------------------------------------------
    // 以下几何参数取自 rm_auto_aim 的 NumberClassifier::extractNumbers：
    //   灯条在 28 像素高的画布中只占 12 像素（上下各留 8 像素给数字），
    //   小装甲板画布宽 32 / 大装甲板 54，最后裁中间 20 列，
    //   得到 20(宽) x 28(高) 的 ROI —— 正好是 mlp.onnx 的输入尺寸。
    //
    // 之前把灯条端点直接拉到整幅画布，等于把灯条当成整块板高，
    // 数字上下被裁掉、宽高也被拉伸，网络必然认不出。
    // ============================================================
    constexpr int kLightLength = 12;   // 灯条在画布中的高度（像素）
    constexpr int kWarpHeight  = 28;   // 画布高度
    constexpr int kSmallWidth  = 32;   // 小装甲板画布宽度
    constexpr int kLargeWidth  = 54;   // 大装甲板画布宽度
    constexpr int kRoiW = 20, kRoiH = 28;

    // src 顺序：左下、左上、右上、右下（与灯条方向一致）
    const cv::Point2f src[4] = {armor.corners[3], armor.corners[0],
                                armor.corners[1], armor.corners[2]};

    const int top_y = (kWarpHeight - kLightLength) / 2 - 1;   // = 7
    const int bot_y = top_y + kLightLength;                   // = 19
    const int warp_w = armor.is_large ? kLargeWidth : kSmallWidth;

    const cv::Point2f dst[4] = {{0.f, static_cast<float>(bot_y)},
                                {0.f, static_cast<float>(top_y)},
                                {static_cast<float>(warp_w - 1), static_cast<float>(top_y)},
                                {static_cast<float>(warp_w - 1), static_cast<float>(bot_y)}};

    const cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::Mat warped;
    cv::warpPerspective(bgr, warped, M, cv::Size(warp_w, kWarpHeight));

    // 裁中间 20 列，保留整幅高度
    const cv::Mat roi =
        warped(cv::Rect(cv::Point((warp_w - kRoiW) / 2, 0), cv::Size(kRoiW, kRoiH)));

    // ---- 二值化：数字是深底亮字，Otsu 即可 ----
    cv::Mat gray, bin;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // ---- 连通域清理：只保留最大的白色块 ----
    // 装甲板 ROI 内，数字是所有白色连通域中最大的；边缘粘连的杂光、
    // 背景碎片会被一并清除。这里是"像素级选择"，不做任何形态学运算，
    // 所以数字的笔画形状完全不会被改变（开运算会腐蚀细笔画）。
    //
    // 保护条件：最大块必须占全部白像素的 60% 以上才执行清理。
    // 否则说明数字被打断（如"5"的上横分离、模糊断笔），
    // 此时宁可不处理，避免把数字的一部分当成杂块删掉。
    {
        cv::Mat labels, stats, centroids;
        const int n = cv::connectedComponentsWithStats(
            bin, labels, stats, centroids, 8, CV_32S);
        if (n > 1) {
            int best = -1, best_area = 0, total_white = 0;
            for (int k = 1; k < n; ++k) {          // k=0 是背景
                const int area = stats.at<int>(k, cv::CC_STAT_AREA);
                total_white += area;
                if (area > best_area) { best_area = area; best = k; }
            }
            if (best > 0 && total_white > 0 &&
                static_cast<double>(best_area) / static_cast<double>(total_white) >= 0.60) {
                cv::Mat cleaned = cv::Mat::zeros(bin.size(), CV_8U);
                cleaned.setTo(255, labels == best);
                bin = cleaned;
            }
        }
    }

    // 调试：设置环境变量 RM_DUMP_ROI=1 时把每块板的 ROI 存下来
    if (std::getenv("RM_DUMP_ROI") != nullptr) {
        static int dump_idx = 0;
        char name[128];
        std::snprintf(name, sizeof(name), "roi_dbg_%02d.png", dump_idx++);
        cv::Mat big;
        cv::resize(bin, big, cv::Size(20 * 8, 28 * 8), 0, 0, cv::INTER_NEAREST);
        cv::imwrite(name, big);
        cv::imwrite(std::string("roi_dbg_col_") + name, roi);
        std::cerr << "[roi] " << name
                  << " TL(" << armor.corners[0].x << "," << armor.corners[0].y
                  << ") TR(" << armor.corners[1].x << "," << armor.corners[1].y
                  << ") BR(" << armor.corners[2].x << "," << armor.corners[2].y
                  << ") BL(" << armor.corners[3].x << "," << armor.corners[3].y
                  << ") large=" << (armor.is_large ? 1 : 0) << "\n";
    }

    // ---- 归一化到 0~1 并组装 blob（不做缩放，尺寸已是 20x28）----
    cv::Mat blob;
    cv::dnn::blobFromImage(bin, blob, 1.0 / 255.0);
    net_.setInput(blob);
    cv::Mat out = net_.forward();   // [1,9] 的原始 logits（模型里没有 Softmax）

    if (out.total() != labels_.size()) return false;

    // ---- 4) softmax 得到概率，再取最大类 ----
    // 模型输出是 logits（图里没有 Softmax 层），这里手动做数值稳定的 softmax。
    cv::Mat flat = out.reshape(1, 1);
    if (!flat.isContinuous()) flat = flat.clone();

    double max_logit = 0.0;
    cv::minMaxLoc(flat, nullptr, &max_logit);

    cv::Mat prob = flat - cv::Scalar(max_logit);   // 减去最大值，防止 exp 溢出
    cv::exp(prob, prob);
    const double sum = cv::sum(prob)[0];
    if (sum <= 0.0) return false;
    prob /= sum;

    cv::Point maxLoc;
    double maxVal = 0.0;
    cv::minMaxLoc(prob, nullptr, &maxVal, nullptr, &maxLoc);

    armor.number = maxLoc.x;
    armor.digit_conf = static_cast<float>(maxVal);
    armor.number_label = labels_[static_cast<size_t>(maxLoc.x)];

    // 置信度不足，或网络判为 negative（非装甲板），视为识别失败
    if (maxVal < min_conf_) return false;
    if (armor.number_label == "negative") return false;

    return true;
}

}  // namespace rm
