#include "evaluator.hpp"

#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace rm {

namespace {

std::string trim(const std::string& s) {
    const char* ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    const size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

}  // namespace

std::vector<TruthPose> Evaluator::loadTruth(const std::filesystem::path& md_path) {
    std::vector<TruthPose> out;
    std::ifstream ifs(md_path);   // C++17：ifstream 直接接受 path，中文路径也安全
    if (!ifs) return out;

    bool in_section = false;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!in_section) {
            // 定位到 "### 1.1 相机光学系下" 这一段
            if (line.find("1.1") != std::string::npos &&
                line.find("相机光学系") != std::string::npos) {
                in_section = true;
            }
            continue;
        }
        // 结束条件必须足够"具体"：不能只判断是否含 "1.2"。
        // 数据行里的数字（如 -51.243、+1.281592）本身就含 "1.2" 子串，
        // 只按 "1.2" 断句会把表格腰斩。这里要求同时含 "世界系"。
        if (line.find("1.2") != std::string::npos &&
            line.find("世界系") != std::string::npos) {
            break;
        }
        if (line.find("armor_") == std::string::npos) continue;

        // 只解析表格行（以 '|' 开头），避免正文里出现 armor_ 时误判
        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] != '|') continue;

        // 按 '|' 切分表格行
        std::vector<std::string> cells;
        std::string cur;
        for (char ch : line) {
            if (ch == '|') { cells.push_back(cur); cur.clear(); }
            else cur.push_back(ch);
        }
        cells.push_back(cur);
        if (cells.size() < 6) continue;

        TruthPose tp;
        tp.name = trim(cells[1]);
        tp.front_visible = (trim(cells[2]) == "正面");
        try {
            tp.t[0] = std::stod(trim(cells[3]));
            tp.t[1] = std::stod(trim(cells[4]));
            tp.t[2] = std::stod(trim(cells[5]));
        } catch (...) {
            continue;   // 表头或格式异常行，跳过
        }
        if (tp.name.empty()) continue;
        out.push_back(tp);
    }
    return out;
}

const TruthPose* Evaluator::pickTruth(const std::vector<TruthPose>& truth,
                                      const cv::Vec3d& measured) {
    const TruthPose* best = nullptr;
    double best_d = 1e18;

    // 第一轮：只在"正面可见"的板里找最近的一块
    for (const auto& t : truth) {
        if (!t.front_visible) continue;
        const double d = cv::norm(t.t - measured);
        if (d < best_d) { best_d = d; best = &t; }
    }
    if (best) return best;

    // 第二轮：没有正面标注时退化为全体里找最近
    for (const auto& t : truth) {
        const double d = cv::norm(t.t - measured);
        if (d < best_d) { best_d = d; best = &t; }
    }
    return best;
}

ErrorRow Evaluator::compare(const std::string& image,
                            const Armor& armor,
                            const std::vector<TruthPose>& truth) {
    ErrorRow row;
    row.image = image;
    row.measured = armor.tvec;

    const TruthPose* tp = pickTruth(truth, armor.tvec);
    if (!tp) return row;

    row.armor = tp->name;
    row.truth = tp->t;
    // 作业要求：Δ = 计算值 - 真值
    row.delta = armor.tvec - tp->t;
    row.err3d = cv::norm(row.delta);
    row.ok = true;
    return row;
}

bool Evaluator::writeCsv(const std::string& path, const std::vector<ErrorRow>& rows) {
    std::ofstream ofs(path);
    if (!ofs) return false;

    ofs << "image,armor,"
           "x_meas,y_meas,z_meas,"
           "x_truth,y_truth,z_truth,"
           "dx,dy,dz,err3d_m\n";

    for (const auto& r : rows) {
        if (!r.ok) {
            ofs << r.image << ",NA,,,,,,,,,\n";
            continue;
        }
        ofs << std::fixed << std::setprecision(6)
            << r.image << "," << r.armor << ","
            << r.measured[0] << "," << r.measured[1] << "," << r.measured[2] << ","
            << r.truth[0] << "," << r.truth[1] << "," << r.truth[2] << ","
            << r.delta[0] << "," << r.delta[1] << "," << r.delta[2] << ","
            << r.err3d << "\n";
    }
    return true;
}

}  // namespace rm
