/**
 * @file wifi_stream.h
 * @brief WiFi station mode and TCP streaming (IMG4 protocol).
 *
 * Streams JPEG frames with exploration metadata to the laptop.
 */
#ifndef WIFI_STREAM_H
#define WIFI_STREAM_H

#include "esp_err.h"
#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Initialize WiFi in station mode and connect to AP.
 * @return ESP_OK on success.
 */
esp_err_t wifi_init_sta(void);

/**
 * @brief Start a TCP server on the given port (non-blocking).
 * Only binds and listens — does NOT block waiting for a client.
 * Call stream_accept_start() afterwards to accept clients in the background.
 */
esp_err_t stream_server_start(uint16_t port);

/**
 * @brief Start background task that accepts TCP clients.
 * Clients can connect/disconnect at any time without affecting the main loop.
 */
void stream_accept_start(void);

/**
 * @brief Send a JPEG frame + exploration metadata to the laptop.
 *
 * Packet format (v4 — free exploration):
 *   [0..3]   magic:        0x494D4734 ("IMG4")
 *   [4..7]   frame_id      (uint32, LE, monotonic)
 *   [8..11]  timestep      (uint32, LE, index within session)
 *   [12]     action_taken  (uint8, 0=FWD, 1=TURN_R, 2=TURN_L, 3=STOP)
 *   [13]     depth_blocked (uint8, 0=clear, 1=blocked)
 *   [14..17] heading_deg   (float32, LE, 0-360 from IMU)
 *   [18..21] jpeg_len      (uint32, LE)
 *   [22..N]  jpeg_data     (variable length)
 *
 * @return ESP_OK on success, ESP_FAIL if client disconnected.
 */
esp_err_t stream_send_frame(uint32_t frame_id, uint32_t timestep,
                            uint8_t action_taken, uint8_t depth_blocked,
                            float heading_deg,
                            const uint8_t *jpeg_buf, uint32_t jpeg_len);

/**
 * @brief Check if a client is currently connected.
 */
bool stream_is_connected(void);

/** Packet magic number "IMG4" (v4 — free exploration) */
#define STREAM_MAGIC  0x494D4734

#endif /* WIFI_STREAM_H */
