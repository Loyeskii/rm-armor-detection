# RM 装甲板识别与位置解算

用 **C++ + OpenCV** 完成装甲板识别与位置解算，使用**传统视觉**与**神经网络**两种方案各做一遍，并对同一批素材给出误差对比。

本仓库包含两套独立工程：

```
02_Recognition/
├── A_conventional/       传统视觉方案：检测识别灯条 + 神经网络识别数字
├── B_neuro_network/      神经网络方案：YOLOv8-pose 检测网络
├── data/
│   ├── original/         素材（A/B 共用）
│   │   ├── images/       三张图 + 真值 md
│   │   └── videos/       测试视频（未上传，需放入测试视频）
│   └── processed/A/      A 的输出：标注图、误差表、标注视频
└── 重投影误差.txt         相机内参与畸变系数
```

> 两个子工程各有 `CMakeLists.txt`、`include/`、`src/`、`config/`、`models/`，可单独构建运行。
> 素材统一放在根目录 `data/`，由配置里的相对路径 `../data/...` 引用。

---

## 快速开始

两个方案步骤一致，以 A 为例（详见各自的 `README.md`）：

```bat
cd A_conventional

:: 构建（需 OpenCV 4.10 + VS2022 + CMake）
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR=C:/opencv/build
cmake --build build --config Release

:: 运行
cd build\Release
armor_traditional.exe --images    :: 图片：识别 + 解算 + 与真值比对
armor_traditional.exe --video     :: 视频：逐帧画框
```

---

## 方案 A 结果概览

三张仿真图（`require_digit: 1`，数字识别失败即丢弃该装甲板）：

| 图片 | 数字识别 | 置信度 | 三维位置误差 \|3D\| |
|---|---|---|---|
| 1.jpg | 3 | 0.977 | 0.114 m |
| 2.jpg（左板） | 3 | 0.944 | 0.423 m |
| 2.jpg（右板） | 3 | 0.731 | 0.924 m |
| 3.jpg | 3 | 0.999 | 0.101 m |

视频：531 帧、1440×1080、30 fps，累计识别 360 次。

完整误差表见 `data/processed/A/images/A_traditional_error.csv`（运行后自动生成）。

---

## 素材说明

| 素材 | 位置 | 是否入库 |
|---|---|---|
| 三张原始图 + 真值 md | `data/original/images/` | ✅ |
| 相机内参 / 畸变 | `重投影误差.txt` | ✅ |
| 测试视频（64 MB） | `data/original/videos/2026-05-20_23-57-31.avi` | 未上传，需自行放入测试视频 |
| 数字网络 `mlp.onnx` | `A_conventional/models/` | ✅ 314 KB |
| 装甲板检测网络 `Infantry-v8n` | `B_neuro_network/models/` | ✅ 6.3 MB |

---

## 环境

- Windows 10/11 x64，MSVC 19.42（VS 2022 BuildTools），CMake ≥ 3.16
- OpenCV **4.10.0**（Windows 预编译包，`vc16`），解压到 `C:\opencv`
- 仅依赖 OpenCV：core / imgproc / imgcodecs / videoio / highgui / calib3d / dnn