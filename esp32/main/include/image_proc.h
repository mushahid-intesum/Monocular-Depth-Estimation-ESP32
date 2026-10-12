/**
 * @file image_proc.h
 * @brief Image processing: downscale and format conversion.
 */
#ifndef IMAGE_PROC_H
#define IMAGE_PROC_H

#include <stdint.h>

/** Target image dimensions for the navigation model. */
#define IMG_TARGET_W    96
#define IMG_TARGET_H    96
#define IMG_TARGET_CH   3
#define IMG_TARGET_SIZE (IMG_TARGET_W * IMG_TARGET_H * IMG_TARGET_CH)

/**
 * @brief Downsample RGB565 image to 96x96 RGB888.
 *
 * Uses area-averaging (box filter) for anti-aliased downscaling.
 * The 160° fisheye distortion is intentionally preserved.
 *
 * @param[in]  src_rgb565  Source buffer in RGB565 format.
 * @param[in]  src_w       Source width (e.g., 320).
 * @param[in]  src_h       Source height (e.g., 240).
 * @param[out] dst_rgb888  Destination buffer, must be at least IMG_TARGET_SIZE bytes.
 */
void image_downsample(const uint8_t *src_rgb565, int src_w, int src_h,
                      uint8_t *dst_rgb888);

/**
 * @brief Convert a single RGB565 pixel to RGB888.
 *
 * @param[in]  pixel  RGB565 value (16-bit, big-endian from camera).
 * @param[out] r      Red channel (0-255).
 * @param[out] g      Green channel (0-255).
 * @param[out] b      Blue channel (0-255).
 */
static inline void rgb565_to_rgb888(uint16_t pixel, uint8_t *r, uint8_t *g, uint8_t *b)
{
    /* OV3660 outputs big-endian RGB565; swap bytes first */
    pixel = (pixel >> 8) | (pixel << 8);
    *r = (uint8_t)((pixel >> 11) & 0x1F) << 3;
    *g = (uint8_t)((pixel >> 5)  & 0x3F) << 2;
    *b = (uint8_t)((pixel)       & 0x1F) << 3;
}

#endif /* IMAGE_PROC_H */
