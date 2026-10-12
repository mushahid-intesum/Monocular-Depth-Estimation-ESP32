# Camera Board — ESP32-S3 Firmware
# Handles: OV3660 camera, depth guard inference, WiFi streaming, SPI master
cmake_minimum_required(VERSION 3.16)

# ── Test mode toggle ──
# Build with: idf.py build -DTEST_MODE=1
# Normal build: idf.py build  (or -DTEST_MODE=0)
option(TEST_MODE "Enable integration test mode" OFF)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(cam_board)

if(TEST_MODE)
    idf_build_set_property(COMPILE_OPTIONS "-DTEST_MODE=1" APPEND)
    message(STATUS "*** TEST MODE ENABLED ***")
endif()
