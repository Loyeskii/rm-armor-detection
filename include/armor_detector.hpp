#pragma once

// 传统视觉装甲板识别：颜色分割 → 找灯条 → 灯条配对。
// 对应理论：HSV 颜色空间分割、形态学、轮廓提取、最小外接矩形、
//           几何约束筛选（长宽比 / 倾角 / 长度比 / 中心距）。
//
// 说明：本仿真素材的灯条是"纯蓝色发光条"，灰度值很低（纯蓝的亮度只有 ~29），
//       用灰度阈值会把灯条漏掉、反而抓到白色数字。因此默认走 HSV 颜色分割；
//       真实相机灯条过曝发白时，可把 config 的 detect.mode 改成 "gray"。

#include "config.hpp"
#include "types.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace rm {

class ArmorDetector {
public:
    explicit ArmorDetector(const Config& cfg) : cfg_(cfg) {}

    // 输入去畸变后的 BGR 图，输出候选装甲板
    // （四点已按 TL,TR,BR,BL 排好；数字与位姿尚未处理）
    std::vector<Armor> detect(const cv::Mat& bgr);

    // 最近一次使用的检测掩码，调试时可 imwrite 出来看
    const cv::Mat& binary() const { return binary_; }

private:
    // 按颜色生成掩码（支持 hsv 区间 / 通道占优两种方式）
    cv::Mat buildColorMask(const cv::Mat& bgr, bool blue) const;
    // 在给定掩码上提灯条；color 直接标记这批灯条的颜色。
    // dom = 该颜色的"通道优势"图（蓝是 B-R，红是 R-B，灰度模式是灰度本身），
    //       仅用于端点精修：在灯条窄带内重新取自适应阈值，剔除两端的光晕细尾。
    std::vector<LightBar> findLights(const cv::Mat& mask, const cv::Mat& dom,
                                     ArmorColor color) const;

    std::vector<Armor> matchLights(const std::vector<LightBar>& lights) const;

    Config cfg_;
    cv::Mat binary_;
};

}  // namespace rm
