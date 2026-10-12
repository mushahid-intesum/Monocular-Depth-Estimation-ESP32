/**
 * @file camera.c
 * @brief OV3660 camera driver with JPEG and RGB565 modes.
 *
 * Based on: espressif/esp32-camera take_picture.c example.
 */
#include "camera.h"
#include "config.h"
#include "esp_log.h"

static const char *TAG = "camera";

/**
 * @brief Internal init with selectable pixel format.
 *
 * Follows the official esp32-camera example config exactly.
 */
static esp_err_t camera_init_common(pixformat_t format)
{
    camera_config_t config = {
        .pin_pwdn     = CAM_PIN_PWDN,
        .pin_reset    = CAM_PIN_RESET,
        .pin_xclk     = CAM_PIN_XCLK,
        .pin_sccb_sda = CAM_PIN_SIOD,
        .pin_sccb_scl = CAM_PIN_SIOC,
        .pin_d7       = CAM_PIN_D7,
        .pin_d6       = CAM_PIN_D6,
        .pin_d5       = CAM_PIN_D5,
        .pin_d4       = CAM_PIN_D4,
        .pin_d3       = CAM_PIN_D3,
        .pin_d2       = CAM_PIN_D2,
        .pin_d1       = CAM_PIN_D1,
        .pin_d0       = CAM_PIN_D0,
        .pin_vsync    = CAM_PIN_VSYNC,
        .pin_href     = CAM_PIN_HREF,
        .pin_pclk     = CAM_PIN_PCLK,

        .xclk_freq_hz  = CAM_XCLK_FREQ,
        .ledc_timer    = LEDC_TIMER_0,
        .ledc_channel  = LEDC_CHANNEL_0,
        .pixel_format  = format,
        .frame_size    = FRAMESIZE_QVGA,    /* 320×240 */
        .jpeg_quality  = CAM_JPEG_QUALITY,
        .fb_count      = CAM_FB_COUNT,
        .fb_location   = CAMERA_FB_IN_PSRAM,
        .grab_mode     = CAMERA_GRAB_WHEN_EMPTY,
    };

    ESP_LOGI(TAG, "Calling esp_camera_init (format=%s)...",
             format == PIXFORMAT_JPEG ? "JPEG" : "RGB565");

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
        ESP_LOGE(TAG, "Check: camera ribbon cable seated correctly?");
        return err;
    }

    /* Sensor tuning */
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        ESP_LOGI(TAG, "Sensor PID: 0x%02x", s->id.PID);
        s->set_brightness(s, 0);
        s->set_contrast(s, 0);
        s->set_saturation(s, 0);
        s->set_whitebal(s, 1);
        s->set_awb_gain(s, 1);
        s->set_exposure_ctrl(s, 1);
        s->set_aec2(s, 1);
        s->set_gain_ctrl(s, 1);
    }

    const char *fmt_str = (format == PIXFORMAT_JPEG) ? "JPEG" : "RGB565";
    ESP_LOGI(TAG, "Camera initialized (QVGA %s)", fmt_str);
    return ESP_OK;
}

esp_err_t camera_init_jpeg(void)
{
    return camera_init_common(PIXFORMAT_JPEG);
}

esp_err_t camera_init_rgb(void)
{
    return camera_init_common(PIXFORMAT_RGB565);
}

camera_fb_t *camera_capture_frame(void)
{
    return esp_camera_fb_get();
}

void camera_release_frame(camera_fb_t *fb)
{
    if (fb) {
        esp_camera_fb_return(fb);
    }
}
