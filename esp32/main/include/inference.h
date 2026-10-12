/**
 * @file inference.h
 * @brief Depth guard + encoder inference on Camera Board.
 *
 * Depth guard:  96×96 RGB → obstacle probability (binary)
 * Encoder:      96×96 RGB → 6×6×1024 feature map (for navigation)
 *
 * Both use TFLite Micro INT8 models.
 */
#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Model dimensions ── */

/** Depth guard input: 96×96×3 = 27648 bytes. */
#define DEPTH_GUARD_INPUT_SIZE  (96 * 96 * 3)

/** Depth guard output: 1 INT8 value (obstacle logit). */
#define DEPTH_GUARD_OUTPUT_SIZE 1

/** Encoder feature output: 6×6×1024 = 36864 bytes. */
#define ENCODER_FEAT_SIZE       (6 * 6 * 1024)

/** TFLite Micro arena sizes — allocated in PSRAM at runtime. */
#define DEPTH_GUARD_ARENA_SIZE  (4864 * 1024)  /* 4.75 MB — 4M param model at 96×96 */
#define ENCODER_ARENA_SIZE      (256 * 1024)   /* 256 KB — placeholder until model ready */

/** Obstacle detection threshold on INT8 logit output.
 *  Corresponds to sigmoid(0) = 0.5 probability.
 *  Positive logit → obstacle detected. */
#define OBSTACLE_LOGIT_THRESHOLD  0

/**
 * @brief Initialize TFLite Micro for depth guard model.
 *
 * Loads the INT8 model from the embedded C array (depth_guard_model.h).
 *
 * @return ESP_OK if model loaded, ESP_ERR_NOT_FOUND if model not embedded.
 */
esp_err_t depth_guard_init(void);

/**
 * @brief Run depth guard: is there an obstacle ahead?
 *
 * Input:  96×96×3 RGB888 image (uint8, will be quantized to int8 internally)
 * Output: true if obstacle detected within ~30cm ahead
 *
 * @param[in]  img_rgb888    96×96×3 uint8 image.
 * @param[out] is_blocked    Set to true if obstacle detected.
 * @return ESP_OK on success, ESP_ERR_INVALID_STATE if not initialized.
 */
esp_err_t depth_guard_run(const uint8_t *img_rgb888, bool *is_blocked);

/**
 * @brief Initialize TFLite Micro interpreter for the encoder.
 * @return ESP_OK on success, ESP_ERR_NOT_FOUND if no model embedded.
 */
esp_err_t inference_init(void);

/**
 * @brief Run the encoder on a 96×96 RGB image.
 *
 * Input:  96×96×3 uint8 image
 * Output: 6×6×1024 int8 feature map
 *
 * @param[in]  img_rgb888    96×96×3 uint8 image.
 * @param[out] features      Output feature map.
 * @return ESP_OK on success.
 */
esp_err_t inference_run_encoder(const uint8_t *img_rgb888, int8_t *features);

/**
 * @brief Run the policy MLP (NOT available on Camera Board).
 * @return ESP_ERR_NOT_SUPPORTED always.
 */
esp_err_t inference_run_policy(const int8_t *corr_cue, int8_t *action_logits);

/**
 * @brief Find the index of the maximum value in an int8 array.
 */
int inference_argmax_i8(const int8_t *arr, int len);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_H */
