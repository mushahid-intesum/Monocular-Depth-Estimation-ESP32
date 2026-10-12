/**
 * @file config.h
 * @brief Hardware configuration for Camera Board (ESP32 #1).
 *
 * Camera pins from Freenove ESP32-S3 WROOM board.
 * SPI master pins for inter-board communication.
 */
#ifndef CONFIG_H
#define CONFIG_H

/* ═══════════════════════════════════════════════════════════════════════════
 *  OV3660 Camera (Freenove ESP32-S3 WROOM)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define CAM_PIN_PWDN    (-1)
#define CAM_PIN_RESET   (-1)
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5
#define CAM_PIN_D7      16
#define CAM_PIN_D6      17
#define CAM_PIN_D5      18
#define CAM_PIN_D4      12
#define CAM_PIN_D3      10
#define CAM_PIN_D2      8
#define CAM_PIN_D1      9
#define CAM_PIN_D0      11
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK    13

#define CAM_XCLK_FREQ   20000000
#define CAM_FB_COUNT     1
#define CAM_JPEG_QUALITY 12

/* ═══════════════════════════════════════════════════════════════════════════
 *  SPI Master — to Motor Board (ESP32 #2)
 *
 *  Using SPI2 (HSPI). Avoid pins used by camera and PSRAM.
 *  GPIOs 33-37 are PSRAM — cannot use.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define SPI_MASTER_MOSI  40
#define SPI_MASTER_MISO  41
#define SPI_MASTER_CLK   42
#define SPI_MASTER_CS    2

#define SPI_CLOCK_HZ     10000000    /* 10 MHz */

/* ═══════════════════════════════════════════════════════════════════════════
 *  Exploration / Streaming
 * ═══════════════════════════════════════════════════════════════════════════ */
#define STREAM_DEFAULT_PORT  8888
#define EXPLORE_CYCLE_MS     1500    /* post-action delay before next capture */

/* ═══════════════════════════════════════════════════════════════════════════
 *  Action Space (shared with Motor Board)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define ACTION_FORWARD    0
#define ACTION_TURN_RIGHT 1
#define ACTION_TURN_LEFT  2
#define ACTION_STOP       3
#define NUM_ACTIONS       4

#endif /* CONFIG_H */
