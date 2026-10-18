cmake_minimum_required(VERSION 3.16)
project(rm_armor_neural LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 源码是 UTF-8，中文 Windows 下 MSVC 默认按 GBK(936) 读，
# 会把中文字符尾字节误当成代码，产生大量假语法错误。这条必须保留。
if(MSVC)
  add_compile_options(/utf-8)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui calib3d dnn)
message(STATUS "OpenCV ${OpenCV_VERSION} @ ${OpenCV_DIR}")

file(GLOB RM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(armor_neural ${RM_SOURCES})

target_include_directories(armor_neural PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(armor_neural PRIVATE ${OpenCV_LIBS})

# 把工程根目录写进程序：运行时用它定位 config/ 与 ../data/
target_compile_definitions(armor_neural PRIVATE
    RM_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

# 构建后把 config/ 拷到 exe 旁边，方便单独拷贝程序运行
add_custom_command(TARGET armor_neural POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/config"
            "$<TARGET_FILE_DIR:armor_neural>/config")

if(WIN32)
  add_custom_command(TARGET armor_neural POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy -t "$<TARGET_FILE_DIR:armor_neural>"
              "$<TARGET_RUNTIME_DLLS:armor_neural>"
      COMMAND_EXPAND_LISTS)

  # videoio 的后端（FFmpeg / MSMF）是运行时按需加载的，不在链接依赖里，
  # 必须单独拷；缺 FFmpeg 时表现为能读 avi、但写 mp4 失败。
  if(DEFINED OpenCV_LIB_PATH)
    file(GLOB _ocv_videoio_dlls "${OpenCV_LIB_PATH}/../bin/opencv_videoio_*_64.dll")
    list(FILTER _ocv_videoio_dlls EXCLUDE REGEX "d\\.dll$")
    if(_ocv_videoio_dlls)
      add_custom_command(TARGET armor_neural POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy -t "$<TARGET_FILE_DIR:armor_neural>"
                  ${_ocv_videoio_dlls}
          COMMAND_EXPAND_LISTS)
    endif()
  endif()
endif()
