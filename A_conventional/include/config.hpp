#pragma once

// 配置读取：把 config.yaml 里的参数装进一个结构体，全项目共享。
// 设计意图：算法里的所有魔法数字都来自配置，而不是硬编码在 .cpp 里。

#include "types.hpp"
#include <string>
#include <vector>

namespace rm {

struct Config {
    CameraIntrinsic camera;

    struct ArmorSize {
        double small_w = 0.135, small_h = 0.125;
        double large_w = 0.225, large_h = 0.125;
    } armor_size;

    struct LightFilter {
        double min_area = 6.0;
        double min_ratio = 1.8;
        double max_ratio = 8.0;
        double max_tilt_deg = 35.0;
        double min_fill = 0.50;   // 轮廓面积 / 最小外接矩形面积

        // 端点精修的阈值比例（0 = 关闭）。
        // 灯条是发光体，轮廓边缘有一圈渐暗的光晕，宽松的颜色掩码会把光晕
        // 也算进轮廓，形成上下各几像素的"锥形细尾"。细尾被当成端点后，
        // 同一块板的两根灯条长度会相差一倍以上，四边形被拉歪、数字 ROI 随之错位。
        // 精修时在灯条自身的窄带内，按该灯条自己的亮度重新取阈值
        // （该区域"颜色优势"值的 90 分位 × 本比例），只在高于阈值的像素上
        // 重新求端点——阈值逐灯条自适应，远处的暗灯条不会被误裁。
        double end_refine = 0.5;

        // 取哪个分位作为"灯条峰值亮度"。灯条沿轴的亮度是渐变的：
        //   p90 → 分界线偏低，光晕尾仍会被算进端点（2.jpg 左板 TL 偏高 18px）
        //   p98 → 分界线过高，远处的暗灯条会被裁掉（1.jpg 直接检不出）
        // 取中间值，兼顾"切干净"和"不误杀"。
        double end_refine_pct = 0.93;
    } light;

    // 检测方式与颜色分割参数
    // 说明：这里用独立标量而不是 [h,s,v] 数组，因为 OpenCV FileStorage
    //       读取序列到 Mat/vector 的行为不稳定，标量最可靠。
    struct Detect {
        std::string mode = "bdiff";  // "bdiff" 通道占优 | "hsv" HSV区间 | "gray" 灰度阈值
        int binary = 40;             // gray 模式使用
        bool use_blue = true;
        bool use_red = false;        // 仿真素材里只有蓝方车
        // bdiff 模式：目标通道比另一通道高出 delta 且自身足够亮
        int bdiff_delta = 12;
        int bdiff_min = 45;
        // hsv 模式（备选）
        int blue_h_min = 100, blue_h_max = 140;
        int red1_h_min = 0,   red1_h_max = 10;
        int red2_h_min = 170, red2_h_max = 180;
        int s_min = 120, v_min = 120;
    } detect;

    struct PairFilter {
        double min_length_ratio = 0.75;
        double min_center_dist = 0.7;
        double max_center_dist = 2.2;
        double min_aspect = 0.6;    // 四边宽高比下限
        double max_aspect = 2.1;    // 四边宽高比上限（剔除"跨两块板"的扁框）
        double max_tilt_diff_deg = 15.0;
        double max_pair_tilt_deg = 35.0;
    } pair;

    struct Threshold {
        int binary = 100;
        float min_digit_conf = 0.3f;
        bool require_digit = true;
        double large_armor_ratio = 3.2;
        double max_range_m = 15.0;
    } thresh;

    // 路径（load() 之后都已解析成绝对路径）
    std::string digit_model;
    std::string label_file;
    std::string video_in;
    std::string video_out;
    std::string images_dir;
    std::string images_out;

    std::vector<std::string> labels;   // labels.txt 的内容

    // 读取 yaml；失败会抛 std::runtime_error
    static Config load(const std::string& yaml_path);

    // 读取 label_file；失败会抛 std::runtime_error
    void loadLabels();

    // 按类别索引取标签文本
    std::string labelOf(int idx) const;
};

}  // namespace rm
