#pragma once

// 数字识别：把装甲板区域拉正 → 二值化 → 送 MLP 推理。
// 对应理论：透视变换（四点映射）、Otsu 大津法二值化、神经网络前向推理。
//
// 该网络在传统方案里承担两个职责：
//   1) 给出装甲板上的号码；
//   2) 参与灯条配对验证 —— 识别不出数字（或判为 negative）的配对直接丢弃，
//      这正是题目要求的"传统方案结合数字网络完成灯条配对"。

#include "types.hpp"
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

namespace rm {

class DigitClassifier {
public:
    DigitClassifier(const std::string& onnx_path,
                    const std::vector<std::string>& labels,
                    float min_conf);

    bool ready() const { return !net_.empty(); }

    // 对装甲板 armor 做数字识别，结果写回 armor.number / number_label / digit_conf。
    // 返回 true 表示得到可信数字（置信度达标且不是 negative 类）。
    bool classify(const cv::Mat& bgr, Armor& armor) const;

private:
    // net_ 用 mutable：classify() 声明为 const（逻辑上不改变对象可见状态），
    // 但 cv::dnn::Net::setInput/forward 是非 const 成员，必须这样绕开。
    mutable cv::dnn::Net net_;
    std::vector<std::string> labels_;
    float min_conf_ = 0.3f;
};

}  // namespace rm
