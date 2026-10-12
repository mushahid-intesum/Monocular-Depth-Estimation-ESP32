/**
 * @file main.c
 * @brief Camera Board main — dual-mode: collection + navigation.
 *
 * Collection mode (default):
 *   capture → depth guard → SPI exchange → stream to laptop
 *
 * Navigation mode (when encoder model is flashed):
 *   capture → downsample → encode → send features via SPI → stream telemetry
 *
 * Mode is selected based on whether the encoder model is embedded.
 */
#include "config.h"
#include "camera.h"
#include "wifi_stream.h"
#include "spi_master.h"

#include "esp_log.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "inference.h"
#include "image_proc.h"
#include "esp_heap_caps.h"

#if defined(TEST_MODE) && TEST_MODE
#include "test_mode.h"
#endif

static const char *TAG = "cam_main";

/* ═══════════════════════════════════════════════════════════════════════════
 *  Collection Mode — depth guard only, no encoder
 * ═══════════════════════════════════════════════════════════════════════════ */

static void collection_task(void *pvParam)
{
    uint32_t frame_id = 0;
    uint32_t timestep = 0;

    ESP_LOGI(TAG, "Collection loop started (step-and-stop, cycle=%d ms)",
             EXPLORE_CYCLE_MS);

    while (1) {
        /* ── Step-and-stop cycle ──
         *
         * The motor board executes actions in blocking 500ms bursts.
         * This camera loop runs in parallel:
         *   1. Capture frame (while motors are stopped from previous step)
         *   2. Run depth guard on the captured frame
         *   3. SPI exchange: send obstacle flag → receive action+heading
         *      Motor board then executes the action (500ms burst)
         *   4. Stream frame + metadata to laptop (if connected)
         *   5. Wait for motor action to complete before next capture
         */

        /* 1. Capture JPEG frame (motors are stopped at this point) */
        camera_fb_t *fb = camera_capture_frame();
        if (!fb) {
            ESP_LOGW(TAG, "Frame capture failed, skipping");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        /* 2. Depth guard — run obstacle detector on downsampled frame */
        uint8_t obstacle_flag = 0;
        {
            static uint8_t *dg_img = NULL;
            if (!dg_img) {
                dg_img = (uint8_t *)heap_caps_malloc(IMG_TARGET_SIZE, MALLOC_CAP_SPIRAM);
                assert(dg_img && "Failed to alloc dg_img in PSRAM");
            }
            image_downsample(fb->buf, fb->width, fb->height, dg_img);

            bool blocked = false;
            depth_guard_run(dg_img, &blocked);
            obstacle_flag = blocked ? 1 : 0;
        }

        /* 3. SPI exchange with Motor Board
         *    → sends obstacle flag
         *    ← receives action_taken + heading_deg
         *    Motor board then executes the action (blocking burst) */
        uint8_t action_taken = ACTION_STOP;
        float heading_deg = 0.0f;

        /* Brief delay to let motor board's SPI slave queue its transaction */
        vTaskDelay(pdMS_TO_TICKS(10));

        esp_err_t spi_ret = spi_exchange_collection(
            obstacle_flag, &action_taken, &heading_deg
        );

        if (spi_ret != ESP_OK) {
            action_taken = ACTION_STOP;
            heading_deg = 0.0f;
        }

        /* 4. Stream IMG4 to laptop (best-effort — skip if no client) */
        if (stream_is_connected()) {
            stream_send_frame(
                frame_id, timestep, action_taken, obstacle_flag,
                heading_deg, fb->buf, fb->len
            );
        }

        camera_release_frame(fb);

        frame_id++;
        timestep++;

        /* 5. Wait for motor action to complete (~500ms burst + settle)
         *    before capturing the next frame */
        vTaskDelay(pdMS_TO_TICKS(EXPLORE_CYCLE_MS));
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Navigation Mode — encoder + SPI features
 * ═══════════════════════════════════════════════════════════════════════════ */

/** Frame buffer for downscaled 96x96 image — allocated in PSRAM. */
static uint8_t *s_img_buf = NULL;

static void navigation_task(void *pvParam)
{
    uint32_t frame_id = 0;
    int nav_steps = 0;

    ESP_LOGI(TAG, "Navigation loop started");

    /* Allocate image buffer in PSRAM */
    if (!s_img_buf) {
        s_img_buf = (uint8_t *)heap_caps_malloc(IMG_TARGET_SIZE, MALLOC_CAP_SPIRAM);
        assert(s_img_buf && "Failed to alloc s_img_buf in PSRAM");
    }

    while (1) {
        /* 1. Capture frame */
        camera_fb_t *fb = camera_capture_frame();
        if (!fb) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        /* 2. Downsample JPEG→RGB565→96x96 RGB888 */
        image_downsample(fb->buf, fb->width, fb->height, s_img_buf);

        /* 3. Run encoder: 96x96 RGB → 6×6×1024 features */
        static int8_t *features = NULL;
        if (!features) {
            features = (int8_t *)heap_caps_malloc(ENCODER_FEAT_SIZE, MALLOC_CAP_SPIRAM);
            assert(features && "Failed to alloc features in PSRAM");
        }
        esp_err_t enc_ret = inference_run_encoder(s_img_buf, features);
        if (enc_ret != ESP_OK) {
            ESP_LOGE(TAG, "Encoder failed");
            camera_release_frame(fb);
            break;
        }

        /* 4. Depth guard obstacle check */
        uint8_t obstacle_flag = 0;
        {
            bool blocked = false;
            depth_guard_run(s_img_buf, &blocked);
            obstacle_flag = blocked ? 1 : 0;
        }

        /* 5. Send features + obstacle to Motor Board via SPI */
        uint8_t action_taken = ACTION_STOP;
        float heading_deg = 0.0f;

        esp_err_t spi_ret = spi_exchange_nav_features(
            obstacle_flag, features, &action_taken, &heading_deg
        );

        if (spi_ret != ESP_OK) {
            ESP_LOGW(TAG, "SPI nav exchange failed");
        }

        /* 6. Stream telemetry to laptop (if connected) */
        if (stream_is_connected()) {
            stream_send_frame(frame_id, nav_steps, action_taken,
                              obstacle_flag, heading_deg,
                              fb->buf, fb->len);
        }

        camera_release_frame(fb);

        /* 7. Log periodically */
        if (nav_steps % 10 == 0) {
            ESP_LOGI(TAG, "Nav step %d: action=%d heading=%.1f obstacle=%d",
                     nav_steps, action_taken, heading_deg, obstacle_flag);
        }

        /* 8. Check termination */
        if (action_taken == ACTION_STOP) {
            ESP_LOGI(TAG, "Policy chose STOP — navigation complete!");
            break;
        }

        frame_id++;
        nav_steps++;

        if (nav_steps >= 200) {
            ESP_LOGW(TAG, "Max nav steps reached");
            break;
        }

        vTaskDelay(pdMS_TO_TICKS(1000 / 5));  /* 5 Hz */
    }

    ESP_LOGI(TAG, "Navigation ended after %d steps", nav_steps);
    vTaskDelete(NULL);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  App Main
 * ═══════════════════════════════════════════════════════════════════════════ */

void app_main(void)
{
    ESP_LOGI(TAG, "╔═══════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   Camera Board — Dual Mode            ║");
    ESP_LOGI(TAG, "╚═══════════════════════════════════════╝");

    /* NVS */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* Network stack */
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    /* Initialize camera */
    ESP_LOGI(TAG, "[1/5] Initializing camera...");
    ESP_ERROR_CHECK(camera_init_jpeg());

    /* Initialize SPI master */
    ESP_LOGI(TAG, "[2/5] Initializing SPI master...");
    ESP_ERROR_CHECK(spi_master_init());

    /* Initialize depth guard */
    ESP_LOGI(TAG, "[3/5] Initializing depth guard...");
    esp_err_t dg_ret = depth_guard_init();
    if (dg_ret == ESP_OK) {
        ESP_LOGI(TAG, "  Depth guard model loaded — obstacle detection active");
    } else {
        ESP_LOGW(TAG, "  Depth guard stub — obstacle detection disabled");
    }

    /* Connect to WiFi */
    ESP_LOGI(TAG, "[4/5] Connecting to WiFi...");
    ESP_ERROR_CHECK(wifi_init_sta());

#if defined(TEST_MODE) && TEST_MODE
    /* ── TEST MODE: connect to test server instead of normal operation ── */
    ESP_LOGI(TAG, "*** TEST MODE ENABLED ***");
    test_mode_run_camera();
    ESP_LOGI(TAG, "Test mode complete — halting.");
    return;
#else
    /* Start TCP server (non-blocking — just bind + listen) */
    ESP_LOGI(TAG, "[5/5] Starting TCP server...");
    ESP_ERROR_CHECK(stream_server_start(STREAM_DEFAULT_PORT));

    /* Start background task to accept TCP clients */
    stream_accept_start();

    /* Try to initialize encoder — determines mode */
    bool nav_mode = false;
    ret = inference_init();
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Encoder model loaded — NAVIGATION mode");
        nav_mode = true;
    } else {
        ESP_LOGI(TAG, "No encoder model — COLLECTION mode");
    }

    /* Start the main task IMMEDIATELY — no waiting for TCP client */
    if (nav_mode) {
        xTaskCreatePinnedToCore(
            navigation_task, "navigation",
            16384, NULL, 5, NULL, 0
        );
    } else {
        xTaskCreatePinnedToCore(
            collection_task, "collection",
            8192, NULL, 5, NULL, 0
        );
    }

    ESP_LOGI(TAG, "Camera Board running (%s mode). "
                   "TCP streaming available when client connects.",
             nav_mode ? "navigation" : "collection");
#endif
}

