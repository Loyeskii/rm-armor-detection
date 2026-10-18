cmake_minimum_required(VERSION 3.16)
project(rm_armor_traditional LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 源码是 UTF-8，而中文 Windows 下 MSVC 默认按 GBK(936) 解析，
# 会把中文字符尾部的字节误当成代码，产生大量莫名其妙的语法错误。
# 这条必须加：强制 MSVC 以 UTF-8 读取源码与字符串字面量。
if(MSVC)
  add_compile_options(/utf-8)
endif()

# 只依赖 OpenCV 的这几个模块：
#   core/imgproc  —— 图像处理（阈值、轮廓、形态学、透视变换）
#   calib3d       —— 去畸变、solvePnP、projectPoints
#   dnn           —— 加载并运行 mlp.onnx
#   videoio       —— 读写视频
#   highgui/imgcodecs —— 显示与图片读写（评估阶段用）
find_package(OpenCV 4 REQUIRED COMPONENTS
    core imgproc imgcodecs videoio highgui calib3d dnn)

message(STATUS "Found OpenCV ${OpenCV_VERSION}")

# 把所有 .cpp 一起编译；新增源文件不需要改 CMakeLists
file(GLOB RM_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(armor_traditional ${RM_SOURCES})

target_include_directories(armor_traditional PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(armor_traditional PRIVATE ${OpenCV_LIBS})

# 把项目根目录写进程序：运行时用它定位 config/ 和 data/，
# 这样可执行文件放在 build/ 里也能找到相对路径的资源。
target_compile_definitions(armor_traditional PRIVATE
    RM_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

# 构建完成后把 config/ 复制到可执行文件旁边，方便单独拷贝程序运行
add_custom_command(TARGET armor_traditional POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/config"
            "$<TARGET_FILE_DIR:armor_traditional>/config")

# Windows 下把 OpenCV 的 dll 一并拷到 exe 旁边，
# 否则运行时会报"找不到 opencv_world4xx.dll"（除非手动把 bin 加进 PATH）
if(WIN32)
  add_custom_command(TARGET armor_traditional POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy -t "$<TARGET_FILE_DIR:armor_traditional>"
              "$<TARGET_RUNTIME_DLLS:armor_traditional>"
      COMMAND_EXPAND_LISTS)

  # OpenCV 的 videoio 后端（FFmpeg / MSMF）是运行时才按需加载的，
  # 不在 TARGET_RUNTIME_DLLS 里，必须单独拷。
  # 缺 opencv_videoio_ffmpeg 时表现为：VideoCapture 能读 avi，
  # 但 VideoWriter 打开 mp4 失败（isOpened() == false），视频写不出来。
  if(DEFINED OpenCV_LIB_PATH)
    file(GLOB _ocv_videoio_dlls "${OpenCV_LIB_PATH}/../bin/opencv_videoio_*_64.dll")
    list(FILTER _ocv_videoio_dlls EXCLUDE REGEX "d\\.dll$")   # 排除 *d.dll 调试版
    if(_ocv_videoio_dlls)
      add_custom_command(TARGET armor_traditional POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy -t "$<TARGET_FILE_DIR:armor_traditional>"
                  ${_ocv_videoio_dlls}
          COMMAND_EXPAND_LISTS)
    endif()
  endif()
endif()
