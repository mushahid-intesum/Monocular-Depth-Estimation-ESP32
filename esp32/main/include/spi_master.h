/**
 * @file spi_master.h
 * @brief SPI master driver for Camera Board → Motor Board communication.
 *
 * Full-duplex SPI at 10 MHz. Each transaction simultaneously:
 *   MOSI (Cam→Mot): obstacle_flag + optional features
 *   MISO (Mot→Cam): heading + action_taken
 */
#ifndef SPI_MASTER_H
#define SPI_MASTER_H

#include "esp_err.h"
#include "inference.h"
#include <stdint.h>
#include <stdbool.h>

/* ─── Collection mode payload ─── */

/** Data sent from Camera Board to Motor Board (MOSI). */
typedef struct __attribute__((packed)) {
    uint8_t  obstacle_flag;        /**< 0=clear, 1=blocked */
    uint8_t  msg_type;             /**< 0=collection, 1=nav_features */
} spi_cam_to_mot_t;

/** Data received from Motor Board (MISO). */
typedef struct __attribute__((packed)) {
    uint8_t  action_taken;         /**< Last action executed (0-3) */
    uint8_t  _pad;
    float    heading_deg;          /**< IMU heading [0, 360) */
} spi_mot_to_cam_t;

/* ─── Navigation mode payload ─── */

/** Navigation mode: features are sent as raw bytes after the 2B header.
 *  Total MOSI = 2 + ENCODER_FEAT_SIZE bytes. */

/**
 * @brief Initialize SPI master bus and device.
 * @return ESP_OK on success.
 */
esp_err_t spi_master_init(void);

/**
 * @brief Collection mode: exchange obstacle flag ↔ heading+action.
 *
 * @param obstacle   0=clear, 1=blocked
 * @param out_action Receives the action Motor Board just executed.
 * @param out_heading Receives IMU heading from Motor Board.
 * @return ESP_OK on success.
 */
esp_err_t spi_exchange_collection(uint8_t obstacle,
                                  uint8_t *out_action,
                                  float *out_heading);

/**
 * @brief Navigation mode: send encoder features + obstacle flag.
 *
 * @param obstacle   0=clear, 1=blocked
 * @param features   6×6×1024 = 36864 bytes of int8 encoder features.
 * @param out_action Receives the action Motor Board chose.
 * @param out_heading Receives IMU heading.
 * @return ESP_OK on success.
 */
esp_err_t spi_exchange_nav_features(uint8_t obstacle,
                                    const int8_t *features,
                                    uint8_t *out_action,
                                    float *out_heading);

#endif /* SPI_MASTER_H */
