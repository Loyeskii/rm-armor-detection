# RM 装甲板识别与位置解算 · 方案 B（神经网络）

用纯 C++ + OpenCV 实现 RoboMaster 装甲板的**识别 → 数字分类 → 位置解算 → 结果展示/误差评估**。
本目录是**神经网络方案**；传统视觉方案见同级目录 `A_conventional/`。

与方案 A 的区别只有前半段：A 靠颜色分割找灯条再配对，B 用 YOLOv8-pose 网络一次前向
同时得到**装甲板四个角点 + 号码类别**。相机模型、PnP 解算、绘制、误差评估四个模块
与 A 共用同一份实现，因此两个方案的坐标约定与误差**可以直接对比**。

---

## 1. 环境要求

| 项 | 版本 / 说明 |
|---|---|
| C++ 标准 | C++17 |
| 编译器 | MSVC 19.42（VS 2022 BuildTools）实测通过；Linux 下 g++ ≥ 9 亦可 |
| CMake | ≥ 3.16 |
| OpenCV | **4.10.0**，需要 `core imgproc imgcodecs videoio highgui calib3d dnn` |
| 平台 | Windows 实测（Linux 需自行确认 FFmpeg 后端可用） |

> **`/utf-8` 必须保留**（`CMakeLists.txt` 已加）。源码是 UTF-8，中文 Windows 下 MSVC
> 默认按 GBK(936) 读取，会把中文字符尾字节误当成代码，产生一堆奇怪的错误。

## 2. 构建

```bat
cd /d <仓库根目录>\B_neuro_network

cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR=C:/opencv/build
cmake --build build --config Release
```

产物：`build\Release\armor_neural.exe`

构建脚本会自动完成两件事：

1. 把 `config/` 拷到 exe 旁边，程序可脱离源码目录单独运行；
2. 把 `opencv_world4100.dll` 与 `opencv_videoio_ffmpeg4100_64.dll` 拷到 exe 旁边。

> 第二个 DLL 容易漏。`videoio` 的 FFmpeg 后端是**运行时按需加载**的，不在链接依赖里；
> 缺它时表现为「能读 .avi、但 `VideoWriter` 打不开 mp4」。

Linux 下去掉 `-G` / `-A`，用 `-DOpenCV_DIR=<opencv>/lib/cmake/opencv4` 即可。

## 3. 运行

```bat
:: 1) 处理三张图片，输出标注图 + 误差表
build\Release\armor_neural.exe --images

:: 2) 处理视频（仿真素材 + 仿真尺寸）
build\Release\armor_neural.exe --video

:: 3) 处理真实比赛录像（必须换视频专用配置，见第 6 节）
build\Release\armor_neural.exe --video --config config\config_video.yaml
```

参数：

| 参数 | 说明 |
|---|---|
| `--video` | 视频模式（默认）。逐帧检测、画框、写 mp4 |
| `--images` | 图片模式。批量处理目录内图片，并与同名真值 md 比对 |
| `--in <路径>` | 覆盖输入（视频文件路径，或图片目录） |
| `--out <路径>` | 覆盖输出（视频文件路径，或图片输出目录） |
| `--config <路径>` | 指定配置文件，默认 `<源码根>/config/config.yaml` |
| `-h` / `--help` | 打印用法 |

## 4. 目录约定

本工程**不含 `data/` 目录**。三组素材统一放在仓库根的 `data/` 下，由配置里的
`../data/...` 相对路径引用，两套方案共用同一份输入。

```
02_Recognition/                 ← 仓库根
├── data/
│   ├── original/               ← 素材唯一来源，A/B 共用
│   │   ├── images/             ← 1.jpg 1.md 2.jpg 2.md 3.jpg 3.md
│   │   └── videos/             ← 2026-05-20_23-57-31.avi（不入库）
│   └── processed/
│       ├── A/                  ← A 的产物
│       └── B/                  ← B 的产物
│           ├── images/         ← 3 张标注图 + B_neural_error.csv
│           └── videos/         ← B_neural_annotated.mp4
├── A_conventional/
└── B_neuro_network/            ← 本工程
    ├── CMakeLists.txt
    ├── config/
    │   ├── config.yaml         ← 仿真图配置（板尺寸 60×56 mm）
    │   ├── config_video.yaml   ← 真实录像配置（板尺寸 135×125 mm）
    │   └── labels.txt          ← 9 类标签
    ├── models/
    │   ├── Infantry-v8n-fp32.onnx    ← 程序实际加载的模型（12.7 MB）
    │   ├── Infantry-v8n-fp16-...onnx  ← 原始模型，保留作参考，程序不用
    ├── include/                ← 只写声明
    └── src/                    ← 只写定义
```

## 5. 各种素材位置

| 素材 | 位置 | 是否入库 |
|---|---|---|
| 三张原始图 + 真值 md | `data/original/images/` | ✅ |
| 测试视频（64 MB） | `data/original/videos/2026-05-20_23-57-31.avi` | ❌ 体积过大，需自行放入 |
| 装甲板检测网络 | `models/Infantry-v8n-fp32.onnx` | ✅ 12.7 MB |
| 相机内参 / 畸变 | 已抄进 `config/config.yaml` 的 `camera` 段；原始出处见仓库根 `重投影误差.txt` | ✅ |

> 真值 md 只在 `--images` 的**误差评估**阶段读取，检测与解算链路完全碰不到它。

## 6. 两份配置的区别

| 项目 | `config.yaml` | `config_video.yaml` |
|---|---|---|
| 用途 | 三张仿真图 | 真实比赛录像 |
| `armor_size.small_w/h` | `0.060 / 0.056` | `0.135 / 0.125` |
| 其余参数 | 完全相同 | 完全相同 |

仿真车的装甲板实测约 **60 mm**，不是真实标准的 135 mm。若换新素材，**最需要确认的是
这一组尺寸**——三个轴的误差若整体等比放大，基本就是这个假设偏了。

## 7. 输出物

| 命令 | 产物 |
|---|---|
| `--images` | `data/processed/B/images/{1,2,3}_annotated.png`（框 + 号码 + 置信度 + 坐标 + 三维坐标轴）<br>`data/processed/B/images/B_neural_error.csv`（Δx/Δy/Δz 与三维误差） |
| `--video` | `data/processed/B/videos/B_neural_annotated.mp4`（1440×1080, 30 fps，逐帧画框） |

CSV 列：`image, armor, x_meas, y_meas, z_meas, x_truth, y_truth, z_truth, dx, dy, dz, err3d_m`
（单位均为米，`d* = 计算值 − 真值`）。

> 本方案对同一张图可能输出**多块板**（画面里本来就有多块），CSV 会为每块各写一行。

## 8. 算法流程与模块对应

```
① 去畸变          camera.cpp          内参按分辨率缩放 + initUndistortRectifyMap / remap
② 预处理          pose_detector.cpp   resize 640×480 → RGB → /255 → HWC 转 CHW
③ 前向推理        pose_detector.cpp   readNetFromONNX + forward → [1, 21, 6300]
④ 解码            pose_detector.cpp   转置成 [6300, 21]，逐候选取类别 / 关键点 / 重排角点
⑤ NMS             pose_detector.cpp   网络 nms=False，自己做贪心抑制
⑥ 位置解算        pnp_solver.cpp      四点 + 板尺寸 → SOLVEPNP_IPPE → rvec / tvec
⑦ 绘制            visualizer.cpp      四边形框 + 号码 + 坐标 + drawFrameAxes
⑧ 误差评估        evaluator.cpp       解析 图N.md 的「1.1 相机光学系下」表格
  流程编排        main.cpp            串起以上各段，处理未检测/解算失败
```

网络输出张量的含义：

```
[1, 21, 6300]
 21   = 4(box) + 9(类别分数) + 8(4 关键点 × x,y)
 6300 = 60×80 + 30×40 + 15×20     ← stride 8 / 16 / 32 三个尺度
```

三个必须知道的实现细节：

- **框不可用**：该模型输出的 `cx,cy,w,h` 实测恒为 `0`，框只能由四个关键点自己算外接矩形。
- **关键点顺序**：网络点序是「左上、左下、右下、右上」，而 PnP / 画框 / 透视变换要求
  「左上、右上、右下、左下」，所以 `detect()` 里做了 `0→TL, 3→TR, 2→BR, 1→BL` 的重排。
  这里顺序错了，位姿一定解错。
- **大小装甲板**：用四个角点算出的宽高比判断，阈值 `large_aspect = 2.5`（实测小板约 1.9、
  大板约 3.2）。早期误用 1.45 时，仿真小板被判成大板，物方宽取 0.225 而非 0.060，
  `z` 直接被放大 3.75 倍。

### 坐标与单位约定

与方案 A 完全一致，两套结果可直接对比：

- **坐标系**：相机光学系，`x` 向右、`y` 向下、`z` 沿光轴指向画面内，原点在光心。
- **单位**：米（`tvec` 的 x/y/z 均为 m）。
- **位置参考点**：装甲板**自身局部坐标系的原点 = 板中心**（不是四角之一，也不是灯条端点）。
- **物方点**：`(±w/2, ±h/2, 0)`，`+X` 沿板宽、`+Y` 沿板高、`+Z` 为板面法线。
- **特征点对应**：`corners[0..3] = 左上、右上、右下、左下`，与物方点一一对应且顺序固定。
- **畸变处理**：先 `initUndistortRectifyMap` + `remap` 去畸变；`newCameraMatrix` 直接取
  `K` 本身，所以后续 PnP 用同一个 `K`，不需要任何坐标换算。

## 9. 关于 `Infantry-v8n-fp32.onnx`

原始模型 `Infantry-v8n-fp16-20260726-D1.8w-B16.onnx` 的 147 个权重全是 **float16**，
且图首尾各有一个 `Cast` 节点做 `fp32 ↔ fp16` 转换。OpenCV 的 DNN 只支持 float32，
**加载即失败**（Python 版 4.9 与 C++ 版 4.10 均如此）。

因此本目录额外提供了一个纯 fp32 版本 `Infantry-v8n-fp32.onnx`，由原始文件转换得到：

1. 删除两个 `Cast` 节点（`graph_input_cast_0` / `graph_output_cast_0`），把上游 `images`
   直接接到主干、主干最后一层直接产出 `output0`；
2. 所有 fp16 initializer 转成 fp32（fp16 的每个有限值都能被 fp32 精确表示，权重本身无损）；
3. 其余节点、属性、opset 原样保留。

**对结果的影响**：权重无损，但计算精度从 fp16 提升为 fp32，输出会有约 `1e-3` 量级的
数值差异（关键点零点几像素、类别分数千分之几）。在 one-hot 置信度 0.9 量级下
`argmax` 不会翻转，对识别结论无影响。原始 fp16 文件保留未删，可随时做对照。

> 若以后要换成 ONNX Runtime 后端，可直接加载原始 fp16 模型，无需这步转换。

## 10. 已知限制

1. **每张图会有第二个检测**。画面里本来就有多块装甲板，网络会把侧向板也检出；侧板以很斜的
   角度看过去，单目单视图下位姿歧义大，误差显著高于正面板（见下表 1.jpg 的 `armor_1`）。
2. **重投影误差不能用来判断检测对错**。四点共面时 IPPE 总能精确拟合，误检的
   `reproj` 依然只有 0.2~0.7 px，它只能当「解算过程是否正常」的体检指标。
3. **单目尺度歧义**。不给真实尺寸，绝对距离就存在整体缩放的不确定性；本方案的
   `z` 误差主要来源是角点回归精度与板尺寸假设，而不是网络本身。
4. **真实录像上的号码命中率低于仿真图**。检测框很准，但号码置信度在运动模糊/遮挡帧会掉。
   视频默认不做号码门控（`min_class_conf = 0.30` 以下才显示 `?`），以保证框的召回。
5. **仅 CPU 推理**。未接 OpenVINO / CUDA 后端；531 帧 1440×1080 视频约十几秒跑完。

## 11. 实测结果

图片（`--images`，`config/config.yaml`）：

| 图片 | 号码 | 置信度 | 解算 z (m) | 真值 z (m) | \|3D\| (m) |
|---|---|---|---|---|---|
| 1.jpg（正面板） | 3 | 0.906 | 3.604 | 3.763 | **0.192** |
| 2.jpg（正面板） | 3 | 0.938 | 1.648 | 1.282 | **0.377** |
| 3.jpg（正面板） | 3 | 0.937 | 1.350 | 1.467 | **0.134** |
| 2.jpg（侧板） | 3 | 0.901 | 2.418 | 1.385 | 1.043 |
| 3.jpg（侧板） | 3 | 0.893 | 2.315 | 1.553 | 0.762 |
| 1.jpg（误检） | 3 | 0.873 | 6.527 | 3.795 | 2.733 |

视频（`--video --config config/config_video.yaml`，真实比赛录像）：531 帧，
**累计识别 1110 次**（平均每帧 2 个），框精确贴合装甲板，画面中号码为 `4`。

完整误差表见 `data/processed/B/images/B_neural_error.csv`（运行后自动生成）。

## 12. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `[warn] 检测网络加载失败` | 确认 `models/Infantry-v8n-fp32.onnx` 存在；不要改配置去指向 fp16 原文件，OpenCV 加载不了 |
| `VideoWriter` 打不开 mp4 | 缺 `opencv_videoio_ffmpeg4100_64.dll`。检查 exe 同目录是否有该文件，或把 `<opencv>/build/x64/vc16/bin` 加入 PATH |
| 启动报「找不到 opencv_world4100.dll」 | 同上，检查 exe 同目录的 DLL 是否被拷全 |
| 配置解析失败 / 字段读成 0 | `config.yaml` 第一行必须是 `%YAML:1.0`；不要写跨行数组，成组数值请拆成独立标量 |
| 号码显示为 `?` | 该候选的类别置信度低于 `thresh.min_class_conf`（默认 0.30） |
| 距离整体偏大 2~3 倍 | 板尺寸用错了：仿真图用 `config.yaml`（60×56 mm），真实录像用 `config_video.yaml`（135×125 mm） |
| 号码全错一位 | 用了 A 的 `label.txt`。B 必须用 `config/labels.txt`（`class N = 数字 N`） |
