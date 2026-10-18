#pragma once

// 全项目共用的数据结构定义。
// 设计意图：让"灯条""装甲板""相机内参""真值"这些概念各有一个明确的类型，
// 各模块之间只通过这些类型交换数据，避免到处传裸 cv::Point。

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace rm {

// 装甲板颜色（红方 / 蓝方）
enum class ArmorColor { RED = 0, BLUE = 1, UNKNOWN = 2 };

// 相机内参 + 畸变系数
// K: 3x3 内参矩阵 (CV_64F)，D: 1x5 畸变 [k1,k2,p1,p2,k3] (CV_64F)
struct CameraIntrinsic {
    cv::Mat K;
    cv::Mat D;
    int calib_width = 0;   // 内参对应的标定分辨率
    int calib_height = 0;
};

// 一根灯条（装甲板左右两条发光条之一）
struct LightBar {
    cv::RotatedRect rect;    // 最小外接矩形（带角度）
    cv::Point2f top;         // 上端中心（图像坐标，y 较小）
    cv::Point2f bottom;      // 下端中心
    ArmorColor color = ArmorColor::UNKNOWN;
    float length = 0.f;      // 长边长度（像素）
    float width = 0.f;       // 短边宽度（像素）
    float tilt_deg = 0.f;    // 长轴与竖直方向的夹角绝对值（度）
};

// 一块装甲板：识别 + 数字 + 位姿的完整结果
struct Armor {
    // 四个角点，顺序固定：左上、右上、右下、左下。
    // 该顺序同时用于 2D 画框、透视变换取数字、以及 PnP 的 2D 点。
    cv::Point2f corners[4];

    ArmorColor color = ArmorColor::UNKNOWN;
    bool is_large = false;

    int number = -1;             // 数字类别索引（对应 labels.txt 的行号），-1 表示未识别
    std::string number_label;    // 数字文本，如 "3"
    float digit_conf = 0.f;

    cv::Vec3d rvec{0, 0, 0};     // 旋转向量（罗德里格斯表示）
    cv::Vec3d tvec{0, 0, 0};     // 平移向量，相机坐标系，单位 m
    bool solved = false;         // 是否解算成功
    double reproj_err = -1.0;    // 重投影误差（像素），用于判断解算质量
};

// 从 图N.md 解析出的单块装甲板真值（相机光学系，单位 m）
struct TruthPose {
    std::string name;          // "armor_1" ...
    bool front_visible = false;// 是否为「正面」（未被车体遮挡）
    cv::Vec3d t{0, 0, 0};      // tx, ty, tz
};

}  // namespace rm
