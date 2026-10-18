#include "armor_detector.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace rm {

namespace {

constexpr double kPi = 3.14159265358979323846;

inline double deg2rad(double d) { return d * kPi / 180.0; }

}  // namespace

std::vector<Armor> ArmorDetector::detect(const cv::Mat& bgr) {
    std::vector<LightBar> lights;
    binary_ = cv::Mat::zeros(bgr.size(), CV_8U);

    // 闭运算核只沿"竖直"方向：灯条是竖条，只需要把竖向的断点连起来。
    // 用 3x3 方形核会同时向左右膨胀，把灯条旁边 2~3 px 外的孤立蓝光也并进
    // 同一个连通域——实测 2.jpg 左侧灯条因此从 x=372 扩张到 x=368，
    // 质心被拽向左，装甲板左上角点直接落到灯条左外侧的暗区里。
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));
    const bool use_hsv = (cfg_.detect.mode == "hsv");
    cv::Mat hsv;
    if (use_hsv) cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    // 端点精修用的"通道优势"图：蓝灯条看 B-R，红灯条看 R-B。
    // 只在灯条自身的窄带内做自适应阈值，把两端的光晕细尾剔掉。
    cv::Mat ch[3];
    cv::split(bgr, ch);
    cv::Mat dom_blue, dom_red;
    cv::subtract(ch[0], ch[2], dom_blue, cv::noArray(), CV_16S);   // B - R
    cv::subtract(ch[2], ch[0], dom_red, cv::noArray(), CV_16S);    // R - B

    // 处理一种颜色：生成掩码 → 形态学 → 提灯条 → 并入结果
    auto addColor = [&](bool blue) {
        cv::Mat mask;
        if (use_hsv) {
            const cv::Scalar lo(blue ? cfg_.detect.blue_h_min : cfg_.detect.red1_h_min,
                                cfg_.detect.s_min, cfg_.detect.v_min);
            const cv::Scalar hi(blue ? cfg_.detect.blue_h_max : cfg_.detect.red1_h_max, 255, 255);
            cv::inRange(hsv, lo, hi, mask);

            if (!blue) {
                // 红色横跨 0°，还要并上第二段
                cv::Mat m2;
                const cv::Scalar lo2(cfg_.detect.red2_h_min, cfg_.detect.s_min, cfg_.detect.v_min);
                const cv::Scalar hi2(cfg_.detect.red2_h_max, 255, 255);
                cv::inRange(hsv, lo2, hi2, m2);
                cv::bitwise_or(mask, m2, mask);
            }
        } else {
            mask = buildColorMask(bgr, blue);
        }

        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::bitwise_or(binary_, mask, binary_);

        const auto v = findLights(mask, blue ? dom_blue : dom_red,
                                  blue ? ArmorColor::BLUE : ArmorColor::RED);
        lights.insert(lights.end(), v.begin(), v.end());
    };

    if (cfg_.detect.mode == "gray") {
        // 灰度模式：真实相机灯条过曝发白时使用
        cv::Mat gray, mask, gray16;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray16, CV_16S);   // 灰度模式下用亮度本身做端点精修的强度图
        cv::threshold(gray, mask, cfg_.detect.binary, 255, cv::THRESH_BINARY);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        binary_ = mask.clone();

        const auto v = findLights(mask, gray16, ArmorColor::UNKNOWN);
        lights.insert(lights.end(), v.begin(), v.end());
    } else {
        if (cfg_.detect.use_blue) addColor(true);
        if (cfg_.detect.use_red)  addColor(false);
    }

    const auto result = matchLights(lights);
    return result;
}

// 通道占优法：目标通道比对立通道高出 delta，且自身亮度足够。
// 相比 HSV 的 S/V 硬阈值，它对"灯条边缘变暗"更宽容，
// 能保住整根灯条而不被削成半截（这对宽高比判据至关重要）。
cv::Mat ArmorDetector::buildColorMask(const cv::Mat& bgr, bool blue) const {
    std::vector<cv::Mat> ch;
    cv::split(bgr, ch);   // ch[0]=B, ch[1]=G, ch[2]=R

    cv::Mat diff;
    if (blue) {
        cv::subtract(ch[0], ch[2], diff, cv::noArray(), CV_16S);   // B - R
    } else {
        cv::subtract(ch[2], ch[0], diff, cv::noArray(), CV_16S);   // R - B
    }

    cv::Mat m1, m2;
    cv::compare(diff, cv::Scalar(cfg_.detect.bdiff_delta), m1, cv::CMP_GT);
    cv::compare(blue ? ch[0] : ch[2], cv::Scalar(cfg_.detect.bdiff_min), m2, cv::CMP_GT);

    cv::Mat mask;
    cv::bitwise_and(m1, m2, mask);
    return mask;
}

std::vector<LightBar> ArmorDetector::findLights(const cv::Mat& mask, const cv::Mat& dom,
                                               ArmorColor color) const {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector<LightBar> lights;

    for (const auto& c : contours) {
        if (c.size() < 6) continue;
        if (cv::contourArea(c) < cfg_.light.min_area) continue;

        // 最小外接矩形：灯条是细长条，用带角度的外接矩形才能同时拿到长度和倾角
        const cv::RotatedRect rr = cv::minAreaRect(c);
        cv::Point2f v[4];
        rr.points(v);

        const float e01 = static_cast<float>(cv::norm(v[1] - v[0]));
        const float e12 = static_cast<float>(cv::norm(v[2] - v[1]));
        const float len = std::max(e01, e12);   // 长边 = 灯条长度
        const float wid = std::min(e01, e12);   // 短边 = 灯条宽度
        if (wid < 1e-3f) continue;

        // 长宽比筛选：灯条一定是细长的
        const float ratio = len / wid;
        if (ratio < static_cast<float>(cfg_.light.min_ratio) ||
            ratio > static_cast<float>(cfg_.light.max_ratio)) continue;

        // 填充率筛选：灯条与旁边杂光粘连后会变成不规则块，
        // 轮廓面积相对外接矩形的占比会明显下降。
        const double fill = cv::contourArea(c) / static_cast<double>(len * wid);
        if (fill < cfg_.light.min_fill) continue;

        // 长轴方向：取较长的那条边，并统一指向图像下方（y 增大方向）
        cv::Point2f dir = (e01 >= e12) ? (v[1] - v[0]) : (v[2] - v[1]);
        if (dir.y < 0.f) dir = -dir;

        // 倾角 = 长轴与竖直方向的夹角
        const float tilt = std::atan2(std::fabs(dir.x), std::fabs(dir.y)) * 180.f /
                           static_cast<float>(kPi);
        if (tilt > static_cast<float>(cfg_.light.max_tilt_deg)) continue;

        // 沿长轴投影，用"端点区域的质心"作为灯条上下端点。
        // 不能直接取投影极值点：灯条可能与旁边的杂光粘连，
        // 极值点会被拽到粘连一侧，导致整块装甲板的框整体偏移。
        const cv::Point2f u = dir / static_cast<float>(cv::norm(dir));

        float tmin = 1e9f, tmax = -1e9f;
        for (const auto& p : c) {
            const cv::Point2f pf(static_cast<float>(p.x), static_cast<float>(p.y));
            const float t = (pf - rr.center).dot(u);
            tmin = std::min(tmin, t);
            tmax = std::max(tmax, t);
        }
        const float span = tmax - tmin;
        if (span < 1.f) continue;

        // ---- 端点精修：只取"灯条本体"所在的连通块 ----
        // 灯条是发光体，周围有一圈渐暗的光晕；宽松掩码会把光晕、以及**旁边
        // 另一块装甲板漏出的杂光**一起框进轮廓。实测 2.jpg 左侧灯条的左上
        // 约 (370,370) 处就有一小片独立蓝光，它把上端点整体拽高了约 18 px，
        // 于是四边形被拉成一高一低，数字 ROI 随之错位。
        //
        // 做法：
        //   1) 在灯条自身的窄带内，按该灯条自己的亮度取自适应阈值（90 分位 × 系数）
        //   2) 只沿"竖直"方向闭运算桥接亮度起伏造成的断裂（绝不做横向膨胀，
        //      否则又会把旁边的杂光并进来）
        //   3) 取包含灯条中心的那个连通块——孤立的杂光块就此被丢弃
        //   4) 在该连通块上求端点
        float t_lo = tmin, t_hi = tmax;
        if (cfg_.light.end_refine > 0.0 && !dom.empty() && dom.size() == mask.size()) {
            const cv::Rect br = rr.boundingRect();
            const int x0 = std::max(0, br.x - 2);
            const int y0 = std::max(0, br.y - 2);
            const int x1 = std::min(dom.cols - 1, br.x + br.width + 1);
            const int y1 = std::min(dom.rows - 1, br.y + br.height + 1);
            const int bw = x1 - x0 + 1, bh = y1 - y0 + 1;

            if (bw >= 3 && bh >= 3) {
                const cv::Mat sub = dom(cv::Rect(x0, y0, bw, bh));   // CV_16S
                const float half_w = wid * 0.6f + 1.0f;

                cv::Mat tt(bh, bw, CV_32F), ss(bh, bw, CV_32F);
                std::vector<float> hist;   // 窄带内的亮度样本，用于取分位数

                for (int j = 0; j < bh; ++j) {
                    for (int i = 0; i < bw; ++i) {
                        const float px = static_cast<float>(x0 + i) - rr.center.x;
                        const float py = static_cast<float>(y0 + j) - rr.center.y;
                        const float t = px * u.x + py * u.y;
                        const float s = -px * u.y + py * u.x;
                        tt.at<float>(j, i) = t;
                        ss.at<float>(j, i) = s;
                        if (t >= tmin - 1.f && t <= tmax + 1.f && std::fabs(s) <= half_w) {
                            hist.push_back(static_cast<float>(sub.at<short>(j, i)));
                        }
                    }
                }

                if (hist.size() >= 16) {
                    // 用高分位代表"灯条峰值亮度"：灯条沿轴的亮度是渐变的，
                    // 分位取低了光晕尾会被算进端点，取高了远处的暗灯条会被裁掉。
                    const double pct = std::min(0.99, std::max(0.5, cfg_.light.end_refine_pct));
                    const size_t k = static_cast<size_t>(pct * (hist.size() - 1));
                    std::nth_element(hist.begin(), hist.begin() + k, hist.end());
                    const float thr =
                        std::max(20.f, static_cast<float>(cfg_.light.end_refine) * hist[k]);

                    cv::Mat bin;
                    cv::threshold(sub, bin, thr, 255, cv::THRESH_BINARY);
                    bin.convertTo(bin, CV_8U);

                    // 桥接核必须极短：实测 1x5 会把灯条上方 3~4 px 外那一段
                    // 独立杂光"桥"回主体，于是"最大连通块"又把杂光包含进来，
                    // 端点仍然落在杂光顶端。1x3 只够补一个像素级的断裂。
                    const cv::Mat vk =
                        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));
                    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, vk);

                    cv::Mat labels, stats, cents;
                    const int n = cv::connectedComponentsWithStats(
                        bin, labels, stats, cents, 8, CV_32S);

                    // 优先取"灯条中心所在"的连通块；中心恰好落在断裂处时退回面积最大块
                    int chosen = -1;
                    const int ccx = static_cast<int>(std::lround(rr.center.x)) - x0;
                    const int ccy = static_cast<int>(std::lround(rr.center.y)) - y0;
                    if (ccx >= 0 && ccx < bw && ccy >= 0 && ccy < bh) {
                        chosen = labels.at<int>(ccy, ccx);
                    }
                    if (chosen <= 0) {
                        int best_area = 0;
                        for (int m = 1; m < n; ++m) {
                            const int area = stats.at<int>(m, cv::CC_STAT_AREA);
                            if (area > best_area) { best_area = area; chosen = m; }
                        }
                    }

                    if (chosen > 0) {
                        float lo = 1e9f, hi = -1e9f;
                        for (int j = 0; j < bh; ++j) {
                            for (int i = 0; i < bw; ++i) {
                                if (labels.at<int>(j, i) != chosen) continue;
                                const float t = tt.at<float>(j, i);
                                lo = std::min(lo, t);
                                hi = std::max(hi, t);
                            }
                        }
                        // 安全阀：精修后长度不得低于原始的一半，否则说明阈值取坏了
                        if (hi > lo && hi - lo >= 0.5f * span) { t_lo = lo; t_hi = hi; }
                    }
                }
            }
        }

        if (t_hi - t_lo < 2.f) continue;   // 精修后过短，视为噪声

        // 端区质心负责"横向"位置（贴合灯条真实走向与倾斜），
        // 纵向位置直接取精修后的 t_lo / t_hi —— 即剔除光晕后的真实端点。
        // 反过来只用中轴线取点会丢失灯条的横向倾斜，实测会让 2.jpg 的
        // 右板从 3(0.97) 退化为 negative。
        const float margin = 0.18f * (t_hi - t_lo);
        cv::Point2f sum_top(0.f, 0.f), sum_bot(0.f, 0.f);
        int n_top = 0, n_bot = 0;
        for (const auto& p : c) {
            const cv::Point2f pf(static_cast<float>(p.x), static_cast<float>(p.y));
            const float t = (pf - rr.center).dot(u);
            if (t < t_lo - 0.5f || t > t_hi + 0.5f) continue;   // 光晕尾不参与质心
            if (t <= t_lo + margin) { sum_top += pf; ++n_top; }
            else if (t >= t_hi - margin) { sum_bot += pf; ++n_bot; }
        }
        if (n_top == 0 || n_bot == 0) continue;

        LightBar lb;
        const cv::Point2f c_top = sum_top / static_cast<float>(n_top);
        const cv::Point2f c_bot = sum_bot / static_cast<float>(n_bot);
        lb.top = c_top + u * (t_lo - (c_top - rr.center).dot(u));
        lb.bottom = c_bot + u * (t_hi - (c_bot - rr.center).dot(u));
        lb.rect = rr;
        lb.length = t_hi - t_lo;
        lb.width = wid;
        lb.tilt_deg = tilt;
        lb.color = color;    // 颜色由所在掩码决定，不需要再逐像素统计

        lights.push_back(lb);
    }

    return lights;
}

std::vector<Armor> ArmorDetector::matchLights(const std::vector<LightBar>& lights) const {
    std::vector<Armor> armors;

    const float sin_max_pair =
        static_cast<float>(std::sin(deg2rad(cfg_.pair.max_pair_tilt_deg)));

    for (size_t i = 0; i < lights.size(); ++i) {
        for (size_t j = i + 1; j < lights.size(); ++j) {
            const LightBar& a = lights[i];
            const LightBar& b = lights[j];

            // (0) 同色检查：灰度模式下颜色是 UNKNOWN（未知），此时不做颜色区分。
            // 早先写成 `a.color != b.color || a.color == UNKNOWN` 会把这个模式
            // 全部跳掉——灰度模式一帧都配不出板。
            if (a.color != ArmorColor::UNKNOWN && a.color != b.color) continue;

            // (1) 长度比：同一块装甲板的两根灯条应基本等长
            const float lr = std::min(a.length, b.length) / std::max(a.length, b.length);
            if (lr < static_cast<float>(cfg_.pair.min_length_ratio)) continue;

            // (2) 倾角差：同一块装甲板的两根灯条应平行
            if (std::fabs(a.tilt_deg - b.tilt_deg) >
                static_cast<float>(cfg_.pair.max_tilt_diff_deg)) continue;

            // (3) 中心距：按灯条平均长度归一化，排除过近/过远
            const cv::Point2f ca = (a.top + a.bottom) * 0.5f;
            const cv::Point2f cb = (b.top + b.bottom) * 0.5f;
            const float dist = static_cast<float>(cv::norm(ca - cb));
            const float mean_len = 0.5f * (a.length + b.length);
            if (mean_len < 1e-3f) continue;
            const float nd = dist / mean_len;
            if (nd < static_cast<float>(cfg_.pair.min_center_dist) ||
                nd > static_cast<float>(cfg_.pair.max_center_dist)) continue;

            // (4) 两灯条中心连线应与灯条长轴近似垂直（夹角接近 90°）
            const cv::Point2f da = a.bottom - a.top;
            const cv::Point2f dc = cb - ca;
            const float na = static_cast<float>(cv::norm(da));
            const float nc = static_cast<float>(cv::norm(dc));
            if (na < 1e-3f || nc < 1e-3f) continue;
            const float cosv = std::fabs(da.dot(dc) / (na * nc));  // = |cos 夹角|
            if (cosv > sin_max_pair) continue;

            // (5) 两根灯条之间不能夹着第三根灯条
            // 注意：数字笔画也会被颜色分割选中（长度通常只有灯条的一半以下），
            // 不能用它来阻断配对，否则真装甲板会被判掉。
            bool blocked = false;
            for (size_t k = 0; k < lights.size() && !blocked; ++k) {
                if (k == i || k == j) continue;
                if (lights[k].color != a.color) continue;
                if (lights[k].length < 0.5f * mean_len) continue;   // 太短的不是灯条
                const cv::Point2f ck = (lights[k].top + lights[k].bottom) * 0.5f;
                const cv::Point2f n(dc.x / nc, dc.y / nc);
                const cv::Point2f rel = ck - ca;
                const float t = rel.dot(n) / nc;
                const float perp = std::fabs(rel.x * (-n.y) + rel.y * n.x);
                if (t > 0.05f && t < 0.95f && perp < mean_len * 0.25f) blocked = true;
            }
            if (blocked) continue;

            // (6) 统一左右顺序后构造四角点：左上 / 右上 / 右下 / 左下
            const LightBar* left = &a;
            const LightBar* right = &b;
            if (ca.x > cb.x) std::swap(left, right);

            Armor armor;
            armor.color = a.color;
            armor.corners[0] = left->top;
            armor.corners[1] = right->top;
            armor.corners[2] = right->bottom;
            armor.corners[3] = left->bottom;

            // (7) 宽高比约束：排除"跨两块装甲板"的扁平框。
            // 真实装甲板 宽:高 ≈ 135:125 = 1.08（小）/ 225:125 = 1.8（大）。
            // 若把两块板外侧的灯条连起来，框会又宽又扁（实测达 2.2~3.1）。
            const float qw = static_cast<float>(cv::norm(armor.corners[1] - armor.corners[0]));
            const float qh = static_cast<float>(cv::norm(armor.corners[3] - armor.corners[0]));
            if (qh < 1e-3f) continue;
            const float asp = qw / qh;
            if (asp < static_cast<float>(cfg_.pair.min_aspect) ||
                asp > static_cast<float>(cfg_.pair.max_aspect)) {
                continue;
            }

            // 用"中心距/灯条长度"区分大小装甲板
            armor.is_large = (nd > static_cast<float>(cfg_.thresh.large_armor_ratio));

            armors.push_back(armor);
        }
    }

    return armors;
}

}  // namespace rm
