/**
 * @file test_mode.h
 * @brief Integration test mode for Camera Board.
 *
 * When TEST_MODE is enabled, app_main skips the normal collection/navigation
 * loop and instead connects to the test server, then runs test commands
 * received over a TCP test-protocol channel.
 *
 * Enable by setting TEST_MODE=1 in config.h or via build flag:
 *   idf.py build -DEXTRA_CFLAGS="-DTEST_MODE=1"
 */
#ifndef TEST_MODE_H
#define TEST_MODE_H

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Run the camera board test suite.
 *
 * Connects to the test server, sends HELLO, then processes test commands.
 * This function does not return until all tests are done or an error occurs.
 *
 * Requires: WiFi initialized, camera initialized, SPI master initialized.
 *
 * @return ESP_OK on success.
 */
esp_err_t test_mode_run_camera(void);

#ifdef __cplusplus
}
#endif

#endif /* TEST_MODE_H */
