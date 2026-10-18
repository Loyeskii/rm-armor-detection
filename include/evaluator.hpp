#pragma once

// 误差评估：解析真值 md，与解算结果对比。
// 对应理论：坐标系统一、参考点对齐、逐轴误差与三维距离误差。

#include "types.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace rm {

struct ErrorRow {
    std::string image;
    std::string armor;      // 匹配到的真值装甲板名，如 armor_1
    cv::Vec3d measured{0, 0, 0};
    cv::Vec3d truth{0, 0, 0};
    cv::Vec3d delta{0, 0, 0};   // 计算值 - 真值
    double err3d = 0.0;         // 三维位置误差 |delta|
    bool ok = false;            // 是否成功匹配到真值
};

class Evaluator {
public:
    // 解析 图N.md 中 "1.1 相机光学系下" 的表格，读出每块板的 tx,ty,tz
    // 用 std::filesystem::path 接收，避免中文路径在 Windows 下被按 ANSI 代码页误解码
    static std::vector<TruthPose> loadTruth(const std::filesystem::path& md_path);

    // 在真值中挑一块用于对比：优先"正面可见"且离解算结果最近的一块。
    // 返回的指针指向 truth 内部元素，调用方保证 truth 生命周期。
    static const TruthPose* pickTruth(const std::vector<TruthPose>& truth,
                                      const cv::Vec3d& measured);

    static ErrorRow compare(const std::string& image,
                            const Armor& armor,
                            const std::vector<TruthPose>& truth);

    static bool writeCsv(const std::string& path, const std::vector<ErrorRow>& rows);
};

}  // namespace rm
