#include "config.hpp"

#include <opencv2/core.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace rm {

namespace {

// 把配置里的相对路径转成绝对路径（相对于项目根目录）
std::string resolvePath(const std::string& p) {
    if (p.empty()) return p;
    // Windows 盘符路径或 Unix 绝对路径，原样返回
    if (p.size() > 1 && (p[1] == ':' || p[0] == '/' || p[0] == '\\')) return p;
    return std::string(RM_PROJECT_ROOT) + "/" + p;
}

std::string trim(const std::string& s) {
    const char* ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    const size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

}  // namespace

Config Config::load(const std::string& yaml_path) {
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("无法打开配置文件: " + yaml_path);
    }

    Config c;

    // ---- 相机 ----
    // 内参拆成标量读取后手工组矩阵：
    // OpenCV 的 FileStorage 把 YAML 序列读成 Mat 并不稳定，标量最可靠。
    cv::FileNode cam = fs["camera"];
    c.camera.calib_width  = static_cast<int>(cam["calib_width"]);
    c.camera.calib_height = static_cast<int>(cam["calib_height"]);

    const double fx = static_cast<double>(cam["fx"]);
    const double fy = static_cast<double>(cam["fy"]);
    const double cx = static_cast<double>(cam["cx"]);
    const double cy = static_cast<double>(cam["cy"]);

    c.camera.K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx,
                                            0.0, fy, cy,
                                            0.0, 0.0, 1.0);
    c.camera.D = (cv::Mat_<double>(1, 5) << static_cast<double>(cam["k1"]),
                                            static_cast<double>(cam["k2"]),
                                            static_cast<double>(cam["p1"]),
                                            static_cast<double>(cam["p2"]),
                                            static_cast<double>(cam["k3"]));

    // ---- 装甲板尺寸 ----
    cv::FileNode ar = fs["armor"];
    c.armor_size.small_w = static_cast<double>(ar["small_width"]);
    c.armor_size.small_h = static_cast<double>(ar["small_height"]);
    c.armor_size.large_w = static_cast<double>(ar["large_width"]);
    c.armor_size.large_h = static_cast<double>(ar["large_height"]);

    // ---- 灯条筛选 ----
    cv::FileNode li = fs["light"];
    c.light.min_area = static_cast<double>(li["min_area"]);
    c.light.min_ratio = static_cast<double>(li["min_ratio"]);
    c.light.max_ratio = static_cast<double>(li["max_ratio"]);
    c.light.max_tilt_deg = static_cast<double>(li["max_tilt_deg"]);
    if (!li["min_fill"].empty()) c.light.min_fill = static_cast<double>(li["min_fill"]);
    if (!li["end_refine"].empty()) c.light.end_refine = static_cast<double>(li["end_refine"]);
    if (!li["end_refine_pct"].empty())
        c.light.end_refine_pct = static_cast<double>(li["end_refine_pct"]);

    // ---- 检测方式与颜色分割 ----
    cv::FileNode dt = fs["detect"];
    if (!dt.empty()) {
        c.detect.mode = static_cast<std::string>(dt["mode"]);
        if (!dt["binary"].empty())   c.detect.binary = static_cast<int>(dt["binary"]);
        if (!dt["use_blue"].empty()) c.detect.use_blue = (static_cast<int>(dt["use_blue"]) != 0);
        if (!dt["use_red"].empty())  c.detect.use_red  = (static_cast<int>(dt["use_red"]) != 0);

        if (!dt["bdiff_delta"].empty()) c.detect.bdiff_delta = static_cast<int>(dt["bdiff_delta"]);
        if (!dt["bdiff_min"].empty())   c.detect.bdiff_min = static_cast<int>(dt["bdiff_min"]);

        if (!dt["blue_h_min"].empty()) c.detect.blue_h_min = static_cast<int>(dt["blue_h_min"]);
        if (!dt["blue_h_max"].empty()) c.detect.blue_h_max = static_cast<int>(dt["blue_h_max"]);
        if (!dt["red1_h_min"].empty()) c.detect.red1_h_min = static_cast<int>(dt["red1_h_min"]);
        if (!dt["red1_h_max"].empty()) c.detect.red1_h_max = static_cast<int>(dt["red1_h_max"]);
        if (!dt["red2_h_min"].empty()) c.detect.red2_h_min = static_cast<int>(dt["red2_h_min"]);
        if (!dt["red2_h_max"].empty()) c.detect.red2_h_max = static_cast<int>(dt["red2_h_max"]);
        if (!dt["s_min"].empty())      c.detect.s_min = static_cast<int>(dt["s_min"]);
        if (!dt["v_min"].empty())      c.detect.v_min = static_cast<int>(dt["v_min"]);
    }

    // ---- 灯条配对 ----
    cv::FileNode pa = fs["pair"];
    c.pair.min_length_ratio = static_cast<double>(pa["min_length_ratio"]);
    c.pair.min_center_dist = static_cast<double>(pa["min_center_dist"]);
    c.pair.max_center_dist = static_cast<double>(pa["max_center_dist"]);
    if (!pa["min_aspect"].empty()) c.pair.min_aspect = static_cast<double>(pa["min_aspect"]);
    if (!pa["max_aspect"].empty()) c.pair.max_aspect = static_cast<double>(pa["max_aspect"]);
    c.pair.max_tilt_diff_deg = static_cast<double>(pa["max_tilt_diff_deg"]);
    c.pair.max_pair_tilt_deg = static_cast<double>(pa["max_pair_tilt_deg"]);

    // ---- 阈值 ----
    cv::FileNode th = fs["thresh"];
    c.thresh.binary = static_cast<int>(th["binary"]);
    c.thresh.min_digit_conf = static_cast<float>(static_cast<double>(th["min_digit_conf"]));
    c.thresh.require_digit = (static_cast<int>(th["require_digit"]) != 0);
    c.thresh.large_armor_ratio = static_cast<double>(th["large_armor_ratio"]);
    c.thresh.max_range_m = static_cast<double>(th["max_range_m"]);

    // ---- 路径 ----
    cv::FileNode md = fs["model"];
    cv::FileNode io = fs["io"];
    c.digit_model = resolvePath(static_cast<std::string>(md["digit_onnx"]));
    c.label_file  = resolvePath(static_cast<std::string>(md["labels"]));
    c.video_in    = resolvePath(static_cast<std::string>(io["video"]));
    c.video_out   = resolvePath(static_cast<std::string>(io["video_out"]));
    c.images_dir  = resolvePath(static_cast<std::string>(io["images_dir"]));
    c.images_out  = resolvePath(static_cast<std::string>(io["images_out"]));

    return c;
}

void Config::loadLabels() {
    labels.clear();
    std::ifstream ifs(label_file);
    if (!ifs) {
        throw std::runtime_error("无法打开标签文件: " + label_file);
    }
    std::string line;
    while (std::getline(ifs, line)) {
        const std::string t = trim(line);
        if (!t.empty()) labels.push_back(t);
    }
    if (labels.empty()) {
        throw std::runtime_error("标签文件为空: " + label_file);
    }
}

std::string Config::labelOf(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(labels.size())) return "";
    return labels[static_cast<size_t>(idx)];
}

}  // namespace rm
