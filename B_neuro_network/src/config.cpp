#include "config.hpp"

#include <opencv2/core.hpp>
#include <fstream>
#include <stdexcept>

namespace rm {

namespace {

// 把配置里的相对路径转成绝对路径（相对于工程根目录）
std::string resolvePath(const std::string& p) {
    if (p.empty()) return p;
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
        throw std::runtime_error("打不开配置文件: " + yaml_path);
    }

    Config c;

    // ---- 相机内参 ----
    // 说明：OpenCV FileStorage 把 YAML 数组读进 Mat 的行为不稳定，
    //       所以配置里把内参拆成独立标量，这里再拼成矩阵。
    const cv::FileNode cam = fs["camera"];
    if (cam.empty()) throw std::runtime_error("配置缺少 camera 段");
    c.camera.calib_width = static_cast<int>(cam["calib_width"]);
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
    const cv::FileNode as = fs["armor_size"];
    if (!as.empty()) {
        c.armor_size.small_w = static_cast<double>(as["small_w"]);
        c.armor_size.small_h = static_cast<double>(as["small_h"]);
        c.armor_size.large_w = static_cast<double>(as["large_w"]);
        c.armor_size.large_h = static_cast<double>(as["large_h"]);
    }

    // ---- 检测网络 ----
    const cv::FileNode dt = fs["detect"];
    if (!dt.empty()) {
        c.detect.model = resolvePath(static_cast<std::string>(dt["model"]));
        if (!dt["input_w"].empty())    c.detect.input_w = static_cast<int>(dt["input_w"]);
        if (!dt["input_h"].empty())    c.detect.input_h = static_cast<int>(dt["input_h"]);
        if (!dt["conf_thres"].empty()) c.detect.conf_thres = static_cast<double>(dt["conf_thres"]);
        if (!dt["nms_thres"].empty())  c.detect.nms_thres = static_cast<double>(dt["nms_thres"]);
        if (!dt["max_dets"].empty())   c.detect.max_dets = static_cast<int>(dt["max_dets"]);
        if (!dt["swap_rb"].empty())    c.detect.swap_rb = (static_cast<int>(dt["swap_rb"]) != 0);
        if (!dt["large_aspect"].empty())
            c.detect.large_aspect = static_cast<double>(dt["large_aspect"]);
    }

    // ---- 阈值 ----
    const cv::FileNode th = fs["thresh"];
    if (!th.empty()) {
        if (!th["max_range_m"].empty())
            c.thresh.max_range_m = static_cast<double>(th["max_range_m"]);
        if (!th["min_class_conf"].empty())
            c.thresh.min_class_conf = static_cast<double>(th["min_class_conf"]);
    }

    // ---- 路径 ----
    const cv::FileNode md = fs["model"];
    const cv::FileNode io = fs["io"];
    c.label_file = resolvePath(static_cast<std::string>(md["labels"]));
    c.video_in   = resolvePath(static_cast<std::string>(io["video"]));
    c.video_out  = resolvePath(static_cast<std::string>(io["video_out"]));
    c.images_dir = resolvePath(static_cast<std::string>(io["images_dir"]));
    c.images_out = resolvePath(static_cast<std::string>(io["images_out"]));

    return c;
}

void Config::loadLabels() {
    labels.clear();
    std::ifstream ifs(label_file);
    if (!ifs) {
        throw std::runtime_error("打不开标签文件: " + label_file);
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

}  // namespace rm
