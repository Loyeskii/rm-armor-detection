# RM 装甲板识别与位置解算 · 方案 A（传统视觉）

用纯 C++ + OpenCV 实现 RoboMaster 装甲板的**识别 → 数字分类 → 位置解算 → 结果展示/误差评估**。
本目录是作业中的**传统视觉方案**；神经网络方案见同级目录 `B_neuro_network/`。

对应作业要求：

| 作业条目 | 本工程对应内容 |
|---|---|
| 1 传统方案结合数字网络完成灯条配对 | `armor_detector` 找灯条并配对，`number_classifier` 用 `mlp.onnx` 识别号码，识别失败即丢弃该配对（`require_digit: 1`） |
| 2 解算位置并说明坐标约定、处理畸变 | `camera` 去畸变、`pnp_solver` 用 `SOLVEPNP_IPPE` 求位姿 |
| 3 视频画框 + 展示数字 | `--video`，输出 `data/processed/videos/A_traditional_annotated.mp4` |
| 4 图片坐标 vs 真值 | `--images`，输出标注图 + `A_traditional_error.csv` |
| 5 模块划分 / CMake / 头实现分离 / 异常处理 | 见“目录约定”与“算法流程与模块对应” |

---

## 1. 环境要求

| 工具 | 版本（本项目实测） | 说明 |
|---|---|---|
| 操作系统 | Windows 10/11 x64 | 也可在 Linux 下构建，仅需改 `OpenCV_DIR` |
| 编译器 | MSVC 19.42（VS 2022 BuildTools） | 需 C++17 |
| CMake | ≥ 3.16（实测 3.29.5） | VS 自带的即可 |
| OpenCV | **4.10.0**（Windows 预编译包，vc16） | 需要 core / imgproc / imgcodecs / videoio / highgui / calib3d / dnn |

> OpenCV 预编译包下载：<https://opencv.org/releases/> → 4.10.0 → Windows，解压到 `C:\opencv`。
> 确认 `C:\opencv\build\x64\vc16\lib\opencv_world4100.lib` 存在即为正确。

**不需要**额外依赖（不依赖 Eigen / yaml-cpp / ONNX Runtime）。

---

## 2. 构建

```bat
cd A_conventional

"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" ^
  -S . -B build -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR=C:/opencv/build

"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" ^
  --build build --config Release
```

产物在 `build\Release\`。构建脚本会自动把 `config/`、`opencv_world4100.dll`
以及 videoio 后端 `opencv_videoio_ffmpeg4100_64.dll` 拷到 exe 旁边，**无需手动配置 PATH**。

> 若 cmake 已在 PATH 中，直接用 `cmake` 即可；Linux 下去掉 `-G` 与 `-A`。

---

## 3. 运行

```bat
cd build\Release

:: 处理图片：识别 + 解算 + 与真值比对，输出标注图与误差表
armor_traditional.exe --images

:: 处理视频：逐帧识别并在画面上画框、标号码与坐标
armor_traditional.exe --video

:: 视频改用“实拍”配置（见第 6 节）
armor_traditional.exe --video --config ..\..\config\config_video.yaml
```

命令行参数：

| 参数 | 含义 |
|---|---|
| `--images` | 处理 `io.images_dir` 下的图片，与同名 `.md` 真值比对 |
| `--video` | 处理 `io.video` 指定的视频 |
| `--config <路径>` | 指定配置文件，默认 `config/config.yaml` |
| `--in <路径>` | 覆盖输入（图片目录或视频文件） |
| `--out <路径>` | 覆盖输出（图片目录或视频文件） |
| `-h` / `--help` | 打印用法 |

---

## 4. 目录约定

```
A_conventional/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── config/
│   ├── config.yaml          ← 默认配置（三张仿真图）
│   ├── config_video.yaml    ← 视频专用配置（真实比赛录像）
│   └── labels.txt           ← 数字网络的 9 个类别
├── models/
│   └── mlp.onnx             ← 数字识别网络（已入库，314 KB）
├── include/                 ← 头文件：只写声明
│   ├── types.hpp            数据结构：LightBar / Armor / CameraParam
│   ├── config.hpp           配置结构与加载
│   ├── camera.hpp           去畸变
│   ├── armor_detector.hpp   灯条检测与配对
│   ├── number_classifier.hpp 数字识别
│   ├── pnp_solver.hpp       位置解算
│   ├── visualizer.hpp       绘制与视频输出
│   └── evaluator.hpp        真值解析与误差统计
├── src/                     ← 实现文件：只写定义
│   └── （与 include/ 一一对应）+ main.cpp
└── data/
    ├── original/
    │   ├── images/          ← 输入：三张图 + 真值 md（已入库）
    │   └── videos/          ← 输入：测试视频（**不入库**，64 MB）
    └── processed/
        ├── images/          ← 输出：标注图 + A_traditional_error.csv
        └── videos/          ← 输出：A_traditional_annotated.mp4
```

---

## 5. 素材放哪里

| 素材 | 放置位置 | 是否入库 | 来源 |
|---|---|---|---|
| 三张原始图 | `data/original/images/1.jpg` `2.jpg` `3.jpg` | ✅ 入库 | 作业材料 |
| 三张图对应的真值 | `data/original/images/1.md` `2.md` `3.md` | ✅ 入库 | 作业材料 `图1/2/3.md`，重命名为同名以便按文件名配对 |
| 测试视频 | `data/original/videos/2026-05-20_23-57-31.avi` | ❌ **不入库**（64 MB） | 作业材料，请自行放入 |
| 数字识别网络 | `models/mlp.onnx` | ✅ 入库（314 KB） | `chenjunnn/rm_auto_aim` → `armor_detector/model/mlp.onnx` |

> 视频缺失时程序会打印 `[error] 打不开视频: <路径>` 并正常退出，不会崩溃。

---

## 6. 两份配置的区别

| 参数 | `config.yaml`（图片） | `config_video.yaml`（视频） | 原因 |
|---|---|---|---|
| `armor.small_width/height` | **0.060 / 0.056 m** | **0.135 / 0.125 m** | 仿真车装甲板实测约 60 mm，不是真实标准 135 mm |
| `detect.use_blue / use_red` | 1 / 1 | 1 / 1 | 两种颜色同时检测；配对要求同色，互不干扰 |
| `thresh.require_digit` | **1** | **0** | 见第 9 节 |

切换方式：`--config config/config_video.yaml`。

---

## 7. 输出物

| 模式 | 输出 |
|---|---|
| `--images` | `data/processed/images/{1,2,3}_annotated.png`（框 + 号码 + 坐标 + 三维坐标轴）<br>`data/processed/images/A_traditional_error.csv`（Δx/Δy/Δz 与三维误差） |
| `--video` | `data/processed/videos/A_traditional_annotated.mp4`（1440×1080, 30 fps，逐帧画框） |

误差表格式（`计算值 − 真值`，单位 m）：

```
image,armor,x_meas,y_meas,z_meas,x_truth,y_truth,z_truth,dx,dy,dz,err3d_m
```

---

## 8. 算法流程与模块对应

```
① 去畸变        camera.cpp             内参按实际分辨率缩放 + initUndistortRectifyMap / remap
② 颜色分割      armor_detector.cpp     通道占优法（B−R 或 R−B）+ 竖直闭运算
③ 找灯条        armor_detector.cpp     轮廓 → 最小外接矩形 → 长宽比 / 倾角 / 填充率筛选
④ 端点精修      armor_detector.cpp     逐灯条自适应阈值 → 只取最大连通块求端点（剔除光晕与杂光）
⑤ 灯条配对      armor_detector.cpp     同色 / 长度比 / 平行度 / 中心距 / 连线⊥长轴 / 中间无遮挡 → 四点 TL,TR,BR,BL
⑥ 数字识别      number_classifier.cpp  透视变换到 32×28 画布 → 裁 20×28 → Otsu → MLP
⑦ 位置解算      pnp_solver.cpp         物方点 (±w/2, ±h/2, 0) → solvePnP(IPPE) → rvec / tvec
⑧ 展示          visualizer.cpp         polylines 画框 + putText 号码/坐标 + drawFrameAxes
⑨ 误差评估      evaluator.cpp          解析 图N.md 的“1.1 相机光学系下”表格 → Δ = 计算 − 真值
```

### 坐标与单位约定（作业第 2 条）

- **坐标系**：相机光学系，`x` 向右、`y` 向下、`z` 沿光轴向前（OpenCV 约定），与真值 md 一致
- **单位**：米（m）
- **参考点**：装甲板**几何中心**（物方点以板中心为原点），与真值 md 的 `tx,ty,tz` 对齐
- **特征点对应顺序**：`corners[0..3] = 左上 / 右上 / 右下 / 左下`，物方点、透视变换目标点、`solvePnP` 输入三处严格同序
- **畸变**：先去畸变，`newCameraMatrix` 取内参本身，因此 PnP 直接使用同一套内参，无需坐标换算

---

## 9. 关于 `require_digit`

图片配置现在是 `require_digit: 1`——数字识别失败就丢弃该装甲板。这是严格要求，能证明“数字网络真正参与了配对”；代价是换成更难的数据时召回会下降。视频配置保留 `require_digit: 0`，因为真实录像里数字网络命中率低。

---

## 10. 已知限制

1. **装甲板尺寸是标定值，不是题给参数。** 三张仿真图反推出板宽约 60 mm（标准 RM 小装甲板是 135 mm）。该值用 1.jpg 标定、2.jpg 与 3.jpg 独立验证，报告中有说明；换一批素材需重新标定。
2. **单目单视图的尺度歧义。** 不给真实尺寸时绝对距离无法唯一确定；斜视角度大的装甲板（如 2.jpg 右板）位姿歧义更明显，`z` 误差偏大。
3. **真实录像的误检。** `config_video.yaml` 面向实拍（灯条为过曝白 / 偏红），背景中的高亮结构（如明亮窗框）偶尔会被误判为灯条。
4. **数字网络对仿真渲染的泛化。** `mlp.onnx` 在真实比赛图像上训练，仿真渲染的字形与之差异较大；ROI 几何（灯条占 12/28 高度、画布宽 32、裁中间 20 列）严格按训练时的预处理实现。
5. **只露出单根灯条时不解算。** 单灯条无法构成四边形，PnP 无解，该候选会被跳过（属于“解算失败”的合法分支）。

---

## 11. 常见问题

| 现象 | 原因与处理 |
|---|---|
| 启动报“找不到 opencv_world4100.dll” | 构建脚本会自动拷贝；若手动移动过 exe，请连同 `build\Release` 下的 dll 一起带走，或把 `C:\opencv\build\x64\vc16\bin` 加入 PATH |
| `[warn] 视频写入器打开失败` | 缺 `opencv_videoio_ffmpeg4100_64.dll`（videoio 后端运行时加载，不在链接依赖里）。重新构建一次即可自动拷贝 |
| 编译出现大量莫名其妙的语法错误 | MSVC 按 GBK 读 UTF-8 源码。`CMakeLists.txt` 已加 `/utf-8`，自行改构建脚本时请保留 |
| `[error] 图片目录不存在` / `[error] 打不开视频` | 素材未按第 5 节放置 |
| `[warn] 数字网络加载失败` | `models/mlp.onnx` 缺失或损坏（正常大小 314,058 字节） |
