/**
 * @file camera.h
 * @brief OV3660 camera initialization and frame capture.
 *
 * Supports two pixel formats:
 *   JPEG:   For streaming to laptop (Phase 1/2 data collection)
 *   RGB565: For on-board inference (Phase 7 navigation)
 */
#ifndef CAMERA_H
#define CAMERA_H

#include "esp_camera.h"
#include "esp_err.h"
#include <stdint.h>

/**
 * @brief Initialize the OV3660 camera in JPEG mode (for streaming).
 * @return ESP_OK on success.
 */
esp_err_t camera_init_jpeg(void);

/**
 * @brief Initialize the OV3660 camera in RGB565 mode (for inference).
 * @return ESP_OK on success.
 */
esp_err_t camera_init_rgb(void);

/**
 * @brief Capture a frame. Returns the camera framebuffer.
 *
 * Caller MUST call camera_release_frame() when done.
 *
 * @return Pointer to camera_fb_t, or NULL on failure.
 */
camera_fb_t *camera_capture_frame(void);

/**
 * @brief Release the frame buffer back to the DMA pool.
 */
void camera_release_frame(camera_fb_t *fb);

#endif /* CAMERA_H */
