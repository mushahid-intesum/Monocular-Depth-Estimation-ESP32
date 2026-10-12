/**
 * @file spi_master.c
 * @brief SPI master driver — Camera Board side.
 *
 * Uses SPI2_HOST (HSPI). Full-duplex transactions exchange data
 * between Camera Board and Motor Board at 10 MHz.
 *
 * Two modes:
 *   Collection: 6-byte transaction (obstacle → heading+action)
 *   Navigation: 65538-byte transaction (obstacle+features → heading+action)
 */
#include "spi_master.h"
#include "config.h"
#include "esp_log.h"
#include "driver/spi_master.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include <string.h>

static const char *TAG = "spi_master";
static spi_device_handle_t s_spi_dev;

esp_err_t spi_master_init(void)
{
    spi_bus_config_t bus_cfg = {
        .mosi_io_num   = SPI_MASTER_MOSI,
        .miso_io_num   = SPI_MASTER_MISO,
        .sclk_io_num   = SPI_MASTER_CLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 66 * 1024,  /* 66 KB for nav features */
    };

    esp_err_t ret = spi_bus_initialize(SPI2_HOST, &bus_cfg, SPI_DMA_CH_AUTO);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Bus init failed: 0x%x", ret);
        return ret;
    }

    spi_device_interface_config_t dev_cfg = {
        .clock_speed_hz = SPI_CLOCK_HZ,
        .mode           = 0,               /* CPOL=0, CPHA=0 */
        .spics_io_num   = SPI_MASTER_CS,
        .queue_size     = 1,
        .flags          = 0,
    };

    ret = spi_bus_add_device(SPI2_HOST, &dev_cfg, &s_spi_dev);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Device add failed: 0x%x", ret);
        return ret;
    }

    ESP_LOGI(TAG, "SPI master initialized (CLK=%d Hz)", SPI_CLOCK_HZ);
    return ESP_OK;
}

esp_err_t spi_exchange_collection(uint8_t obstacle,
                                  uint8_t *out_action,
                                  float *out_heading)
{
    /*
     * Collection transaction: 6 bytes each way.
     * MOSI: [obstacle_flag, msg_type=0, 0, 0, 0, 0]
     * MISO: [action_taken, pad, heading_deg (float32)]
     */
    spi_cam_to_mot_t tx = {
        .obstacle_flag = obstacle,
        .msg_type      = 0,
    };

    /* Pad tx to match rx size */
    uint8_t tx_buf[6];
    memset(tx_buf, 0, sizeof(tx_buf));
    memcpy(tx_buf, &tx, sizeof(tx));

    uint8_t rx_buf[6];
    memset(rx_buf, 0, sizeof(rx_buf));

    spi_transaction_t t = {
        .length    = 6 * 8,              /* bits */
        .tx_buffer = tx_buf,
        .rx_buffer = rx_buf,
    };

    /* Use queued transaction with timeout to avoid blocking forever */
    esp_err_t ret = spi_device_queue_trans(s_spi_dev, &t, pdMS_TO_TICKS(5000));
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "SPI queue failed: 0x%x", ret);
        return ret;
    }

    spi_transaction_t *rtrans;
    ret = spi_device_get_trans_result(s_spi_dev, &rtrans, pdMS_TO_TICKS(5000));
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "SPI transaction timeout: 0x%x", ret);
        return ret;
    }

    /* Parse MISO */
    spi_mot_to_cam_t rx;
    memcpy(&rx, rx_buf, sizeof(rx));
    *out_action  = rx.action_taken;
    *out_heading = rx.heading_deg;

    return ESP_OK;
}

esp_err_t spi_exchange_nav_features(uint8_t obstacle,
                                    const int8_t *features,
                                    uint8_t *out_action,
                                    float *out_heading)
{
    /*
     * Navigation transaction: ENCODER_FEAT_RAW_SIZE+2 bytes MOSI.
     * MOSI: [obstacle_flag, msg_type=1, features[ENCODER_FEAT_SIZE]]
     * MISO: [action_taken, pad, heading_deg, pad...]
     *
     * Both buffers must be same size for full-duplex.
     * Allocated in PSRAM because they are ~65 KB each.
     */
    static uint8_t *tx_buf = NULL;
    static uint8_t *rx_buf = NULL;
    const int buf_size = 2 + ENCODER_FEAT_SIZE;

    if (!tx_buf) {
        tx_buf = (uint8_t *)heap_caps_malloc(buf_size, MALLOC_CAP_SPIRAM);
        rx_buf = (uint8_t *)heap_caps_malloc(buf_size, MALLOC_CAP_SPIRAM);
        if (!tx_buf || !rx_buf) {
            ESP_LOGE(TAG, "Failed to alloc SPI nav buffers");
            return ESP_FAIL;
        }
    }

    memset(tx_buf, 0, buf_size);
    tx_buf[0] = obstacle;
    tx_buf[1] = 1;   /* msg_type = nav features */
    memcpy(&tx_buf[2], features, ENCODER_FEAT_SIZE);

    memset(rx_buf, 0, buf_size);

    spi_transaction_t t = {
        .length    = buf_size * 8,
        .tx_buffer = tx_buf,
        .rx_buffer = rx_buf,
    };

    esp_err_t ret = spi_device_queue_trans(s_spi_dev, &t, pdMS_TO_TICKS(5000));
    if (ret != ESP_OK) return ret;

    spi_transaction_t *rtrans;
    ret = spi_device_get_trans_result(s_spi_dev, &rtrans, pdMS_TO_TICKS(5000));
    if (ret != ESP_OK) return ret;

    /* Parse first 6 bytes of MISO */
    spi_mot_to_cam_t rx;
    memcpy(&rx, rx_buf, sizeof(rx));
    *out_action  = rx.action_taken;
    *out_heading = rx.heading_deg;

    return ESP_OK;
}
