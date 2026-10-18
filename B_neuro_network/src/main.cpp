// 神经网络方案的主程序：流程编排。
// 只负责"把模块串起来"，不含任何算法细节：
//   去畸变 → 检测网络 → 号码 → PnP 解算 → 绘制 → 误差评估

#include "camera.hpp"
#include "config.hpp"
#include "evaluator.hpp"
#include "pnp_solver.hpp"
#include "pose_detector.hpp"
#include "visualizer.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace rm;

namespace {

void printUsage() {
    std::cout
        << "用法:\n"
        << "  armor_neural --video                  处理配置中的视频\n"
        << "  armor_neural --images                 处理配置目录下的图片并与真值比对\n"
        << "  可选: --in <路径>  --out <路径>  --config <路径>\n";
}

// 单帧完整流水线
std::vector<Armor> processFrame(const cv::Mat& undist, const cv::Mat& K,
                                PoseDetector& detector, const PnPSolver& solver,
                                const Config& cfg) {
    std::vector<Armor> out;
    for (auto& a : detector.detect(undist)) {
        // 网络类别 → 号码文本
        if (a.number >= 0 && a.number < static_cast<int>(cfg.labels.size())) {
            a.number_label = cfg.labels[static_cast<size_t>(a.number)];
        }
        // 置信度太低：保留框但标为未知号码，避免显示错误数字
        if (a.digit_conf < static_cast<float>(cfg.thresh.min_class_conf)) {
            a.number_label = "?";
        }

        // 解算失败（相机背后 / 超量程）就跳过，不让程序崩
        if (!solver.solve(a, K)) continue;
        out.push_back(a);
    }
    return out;
}

int runVideo(const Config& cfg, PoseDetector& detector, const PnPSolver& solver,
             const std::string& in, const std::string& out) {
    cv::VideoCapture cap(in);
    if (!cap.isOpened()) {
        std::cerr << "[error] 打不开视频: " << in << "\n";
        return 1;
    }

    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        std::cerr << "[error] 视频没有可读帧\n";
        return 1;
    }

    const Undistorter undist(cfg.camera, frame.size());
    if (!undist.valid()) {
        std::cerr << "[error] 相机内参无效，请检查 config.yaml 的 camera 段\n";
        return 1;
    }
    const cv::Mat K = undist.K();
    std::cout << "[info] 图像 " << frame.cols << "x" << frame.rows
              << " | 去畸变内参 fx=" << K.at<double>(0, 0)
              << " fy=" << K.at<double>(1, 1)
              << " cx=" << K.at<double>(0, 2)
              << " cy=" << K.at<double>(1, 2) << "\n";

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 1.0) fps = 30.0;

    cv::VideoWriter writer;
    if (!out.empty()) {
        const fs::path p(out);
        if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
        writer.open(out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frame.size());
        if (!writer.isOpened()) {
            std::cerr << "[warn] mp4v 编码器不可用，尝试 avc1 ...\n";
            writer.open(out, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, frame.size());
        }
        if (!writer.isOpened()) {
            std::cerr << "[warn] 视频写入器打开失败，将只处理不保存: " << out << "\n";
        } else {
            std::cout << "[info] 输出视频: " << out << "\n";
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    int idx = 0;
    int total_armor = 0;

    while (!frame.empty()) {
        const cv::Mat img = undist.apply(frame);
        const auto armors = processFrame(img, K, detector, solver, cfg);

        cv::Mat show = img.clone();
        for (const auto& a : armors) Visualizer::drawArmor(show, a, K);
        Visualizer::drawHud(show, idx, static_cast<int>(armors.size()));

        total_armor += static_cast<int>(armors.size());
        if (writer.isOpened()) writer.write(show);

        ++idx;
        if (!cap.read(frame)) break;
    }

    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
    std::cout << "[done] 共 " << idx << " 帧, 平均 "
              << (idx > 0 ? static_cast<double>(ms) / idx : 0.0) << " ms/帧"
              << ", 累计识别 " << total_armor << " 次\n";
    return 0;
}

int runImages(const Config& cfg, PoseDetector& detector, const PnPSolver& solver,
              const std::string& dir, const std::string& out_dir) {
    if (!fs::exists(dir)) {
        std::cerr << "[error] 图片目录不存在: " << dir << "\n";
        return 1;
    }
    fs::create_directories(out_dir);

    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            files.push_back(e.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        std::cerr << "[error] 目录里没有图片: " << dir << "\n";
        return 1;
    }
    std::cout << "[info] 图片输入: " << dir << "  共 " << files.size() << " 张\n";
    std::cout << "[info] 图片输出: " << out_dir << "\n";

    std::vector<ErrorRow> rows;

    for (const auto& f : files) {
      try {
        const cv::Mat raw = cv::imread(f);
        if (raw.empty()) continue;

        const Undistorter undist(cfg.camera, raw.size());
        if (!undist.valid()) {
            std::cerr << "[error] 相机内参无效，跳过 " << f << "\n";
            continue;
        }
        const cv::Mat K = undist.K();
        const cv::Mat img = undist.apply(raw);

        const auto armors = processFrame(img, K, detector, solver, cfg);

        cv::Mat show = img.clone();
        for (const auto& a : armors) Visualizer::drawArmor(show, a, K);

        const fs::path p(f);
        const std::string stem = p.stem().string();
        cv::imwrite((fs::path(out_dir) / (stem + "_annotated.png")).string(), show);

        // 找同名真值：1.md 或 图1.md
        // 路径含中文时不能直接用 std::string 构造 fs::path，必须走 u8path
        std::vector<TruthPose> truth;
        const std::vector<fs::path> cand = {
            fs::path(dir) / (stem + ".md"),
            fs::u8path(dir) / fs::u8path("图" + stem + ".md"),
        };
        for (const auto& c : cand) {
            if (fs::exists(c)) {
                truth = Evaluator::loadTruth(c);
                if (!truth.empty()) break;
            }
        }

        std::cout << "\n=== " << p.filename().string() << " ===\n";
        if (armors.empty()) {
            std::cout << "  未识别到装甲板\n";
            continue;
        }

        std::cout << std::fixed << std::setprecision(3);
        for (const auto& a : armors) {
            std::cout << "  [数字 " << (a.number_label.empty() ? "?" : a.number_label)
                      << "  conf=" << a.digit_conf
                      << "]  解算: x=" << a.tvec[0]
                      << " y=" << a.tvec[1]
                      << " z=" << a.tvec[2] << " m"
                      << "  (重投影 " << a.reproj_err << " px)\n";

            if (truth.empty()) continue;

            const ErrorRow r = Evaluator::compare(p.filename().string(), a, truth);
            rows.push_back(r);
            if (r.ok) {
                std::cout << "      真值(" << r.armor << "): x=" << r.truth[0]
                          << " y=" << r.truth[1] << " z=" << r.truth[2] << " m\n"
                          << "      误差: dx=" << r.delta[0]
                          << " dy=" << r.delta[1]
                          << " dz=" << r.delta[2]
                          << "  |3D|=" << r.err3d << " m\n";
            }
        }
        if (truth.empty()) {
            std::cout << "      (未找到配对真值 md，跳过误差对比)\n";
        }
      } catch (const std::exception& e) {
        std::cerr << "[error] 处理失败: " << e.what() << "\n";
      }
    }

    const std::string csv = (fs::path(out_dir) / "B_neural_error.csv").string();
    if (Evaluator::writeCsv(csv, rows)) {
        std::cout << "\n[info] 误差表已写出: " << csv << "\n";
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string mode = "video";
        std::string in, out, cfg_path;

        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&](const char* name) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error(std::string("参数缺少取值: ") + name);
                }
                return argv[++i];
            };

            if (a == "--video")       mode = "video";
            else if (a == "--images") mode = "images";
            else if (a == "--in")     in = next("--in");
            else if (a == "--out")    out = next("--out");
            else if (a == "--config") cfg_path = next("--config");
            else if (a == "-h" || a == "--help") { printUsage(); return 0; }
            else {
                std::cerr << "未知参数: " << a << "\n";
                printUsage();
                return 2;
            }
        }

        if (cfg_path.empty()) {
            cfg_path = std::string(RM_PROJECT_ROOT) + "/config/config.yaml";
        }

        Config cfg = Config::load(cfg_path);
        cfg.loadLabels();

        PoseDetector detector(cfg);
        if (!detector.ready()) {
            std::cerr << "[error] 检测网络不可用，无法运行（请检查 "
                      << cfg.detect.model << "）\n";
            return 1;
        }
        PnPSolver solver(cfg);

        if (mode == "video") {
            return runVideo(cfg, detector, solver,
                            in.empty() ? cfg.video_in : in,
                            out.empty() ? cfg.video_out : out);
        }
        return runImages(cfg, detector, solver,
                         in.empty() ? cfg.images_dir : in,
                         out.empty() ? cfg.images_out : out);

    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }
}
