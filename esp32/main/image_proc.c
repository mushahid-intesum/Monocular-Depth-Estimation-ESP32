/**
 * @file image_proc.c
 * @brief Area-averaged downscaling from 320x240 RGB565 to 96x96 RGB888.
 */
#include "image_proc.h"
#include <string.h>

void image_downsample(const uint8_t *src_rgb565, int src_w, int src_h,
                      uint8_t *dst_rgb888)
{
    /*
     * Area-averaging (box filter) downscale.
     *
     * For a 320x240 → 96x96 mapping:
     *   x scale = 320/96 ≈ 3.33 pixels per output pixel
     *   y scale = 240/96 = 2.5  pixels per output pixel
     *
     * We use integer bin boundaries by mapping each output pixel
     * to a rectangular region in the source image.
     */

    for (int dy = 0; dy < IMG_TARGET_H; dy++) {
        /* Source row range for this output row */
        int sy_start = (dy * src_h) / IMG_TARGET_H;
        int sy_end   = ((dy + 1) * src_h) / IMG_TARGET_H;

        for (int dx = 0; dx < IMG_TARGET_W; dx++) {
            /* Source column range for this output column */
            int sx_start = (dx * src_w) / IMG_TARGET_W;
            int sx_end   = ((dx + 1) * src_w) / IMG_TARGET_W;

            /* Accumulate RGB over the source region */
            uint32_t acc_r = 0, acc_g = 0, acc_b = 0;
            int count = 0;

            for (int sy = sy_start; sy < sy_end; sy++) {
                const uint16_t *row = (const uint16_t *)(src_rgb565 + sy * src_w * 2);
                for (int sx = sx_start; sx < sx_end; sx++) {
                    uint8_t r, g, b;
                    rgb565_to_rgb888(row[sx], &r, &g, &b);
                    acc_r += r;
                    acc_g += g;
                    acc_b += b;
                    count++;
                }
            }

            /* Average and store */
            int out_idx = (dy * IMG_TARGET_W + dx) * 3;
            if (count > 0) {
                dst_rgb888[out_idx + 0] = (uint8_t)(acc_r / count);
                dst_rgb888[out_idx + 1] = (uint8_t)(acc_g / count);
                dst_rgb888[out_idx + 2] = (uint8_t)(acc_b / count);
            } else {
                dst_rgb888[out_idx + 0] = 0;
                dst_rgb888[out_idx + 1] = 0;
                dst_rgb888[out_idx + 2] = 0;
            }
        }
    }
}
