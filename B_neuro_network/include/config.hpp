#pragma once

// 配置结构与加载（神经网络方案）。
//
// 设计原则与方案 A 一致：所有阈值、路径都放在 config.yaml 里，
// 改参数不需要重新编译。字段名刻意与 A 的 pnp_solver 保持一致
// （armor_size / thresh.max_range_m），这样 PnP 模块可以原样复用。

#include "types.hpp"
#include <string>
#include <vector>

namespace rm {

struct Config {
    // ---- 相机 ----
    CameraIntrinsic camera;

    // ---- 装甲板真实尺寸（米）----
    // 物方点以板中心为原点；误差整体等比放大时，多半就是这个假设不准。
    struct ArmorSize {
        double small_w = 0.060;
        double small_h = 0.056;
        double large_w = 0.225;
        double large_h = 0.125;
    } armor_size;

    // ---- 检测网络 ----
    struct Detect {
        std::string model;          // onnx 路径
        int input_w = 640;          // 网络输入宽（与导出时的 imgsz 一致）
        int input_h = 480;
        double conf_thres = 0.25;   // 类别置信度阈值
        double nms_thres = 0.45;    // NMS 的 IoU 阈值
        int max_dets = 20;          // NMS 后最多保留多少个
        bool swap_rb = true;        // 网络用 RGB 训练 → BGR 转 RGB
        double large_aspect = 2.5; // 框宽高比超过该值判为大装甲板
    } detect;

    // ---- 阈值 ----
    struct Thresh {
        double max_range_m = 15.0;    // 解算距离上限，超过视为无效解
        double min_class_conf = 0.30; // 低于该置信度视为"号码不可信"
    } thresh;

    // ---- 标签 ----
    // 类别索引 → 号码文本。实测该模型的类别顺序是 class N = 数字 N，
    // 与方案 A 所用 mlp.onnx 的 label.txt 顺序不同，因此单独一份。
    std::string label_file;
    std::vector<std::string> labels;

    // ---- 输入输出 ----
    std::string video_in;
    std::string video_out;
    std::string images_dir;
    std::string images_out;

    static Config load(const std::string& yaml_path);
    void loadLabels();
};

}  // namespace rm
