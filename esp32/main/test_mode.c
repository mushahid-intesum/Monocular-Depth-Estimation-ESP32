/**
 * @file test_mode.c
 * @brief Camera Board test mode — connects to test server and runs commands.
 *
 * Protocol: Each message has a 6-byte header:
 *   [4B] magic "TST\x01"
 *   [2B] payload_len (uint16 LE)
 *   [N]  payload (UTF-8 JSON)
 *
 * Uses manual JSON formatting — no cJSON dependency required.
 */
#include "test_mode.h"
#include "config.h"
#include "camera.h"
#include "spi_master.h"
#include "wifi_stream.h"

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>

static const char *TAG = "cam_test";

/* ── Test server connection — edit IP here ── */
#define TEST_SERVER_IP   "192.168.68.108"
#define TEST_SERVER_PORT 8888

#define TEST_MAGIC_0 'T'
#define TEST_MAGIC_1 'S'
#define TEST_MAGIC_2 'T'
#define TEST_MAGIC_3 '\x01'

static int s_test_sock = -1;

/* ── Helpers ── */

static esp_err_t send_all(int sock, const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t *)buf;
    size_t off = 0;
    while (off < len) {
        int sent = send(sock, p + off, len - off, 0);
        if (sent < 0) return ESP_FAIL;
        off += sent;
    }
    return ESP_OK;
}

static esp_err_t recv_exact(int sock, void *buf, size_t len)
{
    uint8_t *p = (uint8_t *)buf;
    size_t off = 0;
    while (off < len) {
        int got = recv(sock, p + off, len - off, 0);
        if (got <= 0) return ESP_FAIL;
        off += got;
    }
    return ESP_OK;
}

static esp_err_t test_send_str(const char *json_str)
{
    uint16_t payload_len = (uint16_t)strlen(json_str);
    uint8_t header[6] = {TEST_MAGIC_0, TEST_MAGIC_1, TEST_MAGIC_2, TEST_MAGIC_3, 0, 0};
    memcpy(&header[4], &payload_len, 2);

    esp_err_t ret = send_all(s_test_sock, header, 6);
    if (ret == ESP_OK) {
        ret = send_all(s_test_sock, json_str, payload_len);
    }
    return ret;
}

/**
 * Receive one test protocol message.
 * Returns dynamically allocated JSON string (caller must free), or NULL.
 */
static char *test_recv_str(void)
{
    uint8_t header[6];
    if (recv_exact(s_test_sock, header, 6) != ESP_OK) return NULL;

    /* Verify magic */
    if (header[0] != TEST_MAGIC_0 || header[1] != TEST_MAGIC_1 ||
        header[2] != TEST_MAGIC_2 || header[3] != TEST_MAGIC_3) {
        ESP_LOGE(TAG, "Bad test magic: %02x%02x%02x%02x",
                 header[0], header[1], header[2], header[3]);
        return NULL;
    }

    uint16_t payload_len;
    memcpy(&payload_len, &header[4], 2);

    if (payload_len > 4096) {
        ESP_LOGE(TAG, "Payload too large: %u", payload_len);
        return NULL;
    }

    char *buf = calloc(1, payload_len + 1);
    if (!buf) return NULL;

    if (recv_exact(s_test_sock, buf, payload_len) != ESP_OK) {
        free(buf);
        return NULL;
    }
    buf[payload_len] = '\0';
    return buf;
}

/**
 * Extract the "cmd" value from a JSON string like {"cmd":"xxx", ...}.
 * Returns pointer into the original string (not a copy). Only valid while
 * the source string is alive. Writes a \0 over the closing quote.
 */
static char *json_get_cmd(char *json)
{
    char *p = strstr(json, "\"cmd\"");
    if (!p) return NULL;
    p = strchr(p + 4, ':');
    if (!p) return NULL;
    p = strchr(p, '"');
    if (!p) return NULL;
    p++;  /* skip opening quote */
    char *end = strchr(p, '"');
    if (!end) return NULL;
    *end = '\0';
    return p;
}

/* ── Send IMG4 packet directly to test socket ── */

static esp_err_t send_img4_to_test(uint32_t frame_id, uint32_t timestep,
                                     uint8_t action, uint8_t blocked,
                                     float heading,
                                     const uint8_t *jpeg, uint32_t jpeg_len)
{
    uint8_t header[22];
    uint32_t magic = 0x494D4734;  /* "IMG4" */
    memcpy(&header[0],  &magic, 4);
    memcpy(&header[4],  &frame_id, 4);
    memcpy(&header[8],  &timestep, 4);
    header[12] = action;
    header[13] = blocked;
    memcpy(&header[14], &heading, 4);
    memcpy(&header[18], &jpeg_len, 4);

    esp_err_t ret = send_all(s_test_sock, header, 22);
    if (ret == ESP_OK) {
        ret = send_all(s_test_sock, jpeg, jpeg_len);
    }
    return ret;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Test command handlers
 * ══════════════════════════════════════════════════════════════════════════ */

static void handle_t1_ok(void)
{
    ESP_LOGI(TAG, "T1: Connectivity confirmed by server");
    test_send_str("{\"status\":\"ok\",\"board\":\"camera\"}");
}

static void handle_t2_capture(void)
{
    ESP_LOGI(TAG, "T2: Capturing frame...");

    camera_fb_t *fb = camera_capture_frame();
    if (!fb) {
        ESP_LOGE(TAG, "T2: Camera capture failed!");
        test_send_str("{\"status\":\"fail\",\"error\":\"camera_capture_failed\"}");
        return;
    }

    ESP_LOGI(TAG, "T2: Got frame: %u bytes, sending IMG4...", (unsigned)fb->len);

    /* Send IMG4 packet over the test socket */
    send_img4_to_test(0, 0, ACTION_STOP, 0, 0.0f, fb->buf, fb->len);
    camera_release_frame(fb);

    /* Wait for T2 ACK from server */
    char *ack = test_recv_str();
    if (ack) {
        ESP_LOGI(TAG, "T2: Server ACK received");
        free(ack);
    }
}

static void handle_t3_spi_send(void)
{
    ESP_LOGI(TAG, "T3: Sending dummy SPI exchange to motor board...");

    uint8_t action_out = ACTION_STOP;
    float heading_out = 0.0f;

    esp_err_t ret = spi_exchange_collection(0, &action_out, &heading_out);

    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "T3: SPI exchange OK — action=%d heading=%.1f",
                 action_out, heading_out);
    } else {
        ESP_LOGW(TAG, "T3: SPI exchange failed: 0x%x", ret);
    }
    /* Motor board reports result to server directly */
}

static void handle_t5_full_cycle(void)
{
    ESP_LOGI(TAG, "T5: Full pipeline — capture + SPI + stream...");

    /* 1. Capture frame */
    camera_fb_t *fb = camera_capture_frame();
    if (!fb) {
        ESP_LOGE(TAG, "T5: Capture failed");
        test_send_str("{\"status\":\"fail\",\"error\":\"capture_failed\"}");
        return;
    }

    /* 2. SPI exchange with motor board */
    uint8_t action = ACTION_STOP;
    float heading = 0.0f;
    esp_err_t spi_ret = spi_exchange_collection(0, &action, &heading);

    if (spi_ret != ESP_OK) {
        ESP_LOGW(TAG, "T5: SPI failed, using defaults");
        action = ACTION_STOP;
        heading = 0.0f;
    } else {
        ESP_LOGI(TAG, "T5: SPI OK — action=%d heading=%.1f", action, heading);
    }

    /* 3. Send IMG4 to test server */
    send_img4_to_test(1, 1, action, 0, heading, fb->buf, fb->len);
    camera_release_frame(fb);

    /* 4. Send OK via test channel */
    char resp[128];
    snprintf(resp, sizeof(resp),
             "{\"status\":\"ok\",\"action\":%d,\"heading\":%.1f}",
             action, heading);
    test_send_str(resp);

    ESP_LOGI(TAG, "T5: Full pipeline complete");
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Main test loop
 * ══════════════════════════════════════════════════════════════════════════ */

esp_err_t test_mode_run_camera(void)
{
    ESP_LOGI(TAG, "╔═══════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   Camera Board — TEST MODE            ║");
    ESP_LOGI(TAG, "╚═══════════════════════════════════════╝");

    /* ── Connect to test server ── */
    ESP_LOGI(TAG, "Connecting to test server %s:%d...", TEST_SERVER_IP, TEST_SERVER_PORT);

    struct sockaddr_in server_addr = {
        .sin_family = AF_INET,
        .sin_port   = htons(TEST_SERVER_PORT),
    };
    inet_aton(TEST_SERVER_IP, &server_addr.sin_addr);

    /*
     * TCP connect retry: must create a fresh socket per attempt.
     * Once connect() fails on a stream socket, POSIX leaves it in
     * an undefined state — you cannot retry on the same fd.
     */
    bool connected = false;
    for (int i = 0; i < 15; i++) {
        s_test_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (s_test_sock < 0) {
            ESP_LOGE(TAG, "Socket creation failed: errno=%d", errno);
            vTaskDelay(pdMS_TO_TICKS(2000));
            continue;
        }

        int ret = connect(s_test_sock, (struct sockaddr *)&server_addr, sizeof(server_addr));
        if (ret == 0) {
            connected = true;
            ESP_LOGI(TAG, "Connected to test server!");
            break;
        }

        ESP_LOGW(TAG, "Connect attempt %d/15 failed: errno=%d (%s)",
                 i + 1, errno,
                 errno == 111 ? "ECONNREFUSED — server not running?" :
                 errno == 113 ? "EHOSTUNREACH — wrong IP?" :
                 errno == 110 ? "ETIMEDOUT — firewall?" : "unknown");
        close(s_test_sock);
        s_test_sock = -1;
        vTaskDelay(pdMS_TO_TICKS(2000));
    }

    if (!connected) {
        ESP_LOGE(TAG, "╔═══════════════════════════════════════╗");
        ESP_LOGE(TAG, "║  FAILED to connect to test server!    ║");
        ESP_LOGE(TAG, "║  Check:                               ║");
        ESP_LOGE(TAG, "║  1. Is test_server.py running?        ║");
        ESP_LOGE(TAG, "║  2. Is IP correct? (%s)     ", TEST_SERVER_IP);
        ESP_LOGE(TAG, "║  3. Firewall open on port %d?      ", TEST_SERVER_PORT);
        ESP_LOGE(TAG, "║  4. Same WiFi network?                ║");
        ESP_LOGE(TAG, "╚═══════════════════════════════════════╝");
        return ESP_FAIL;
    }

    /* ── Send HELLO ── */
    test_send_str("{\"board\":\"camera\",\"status\":\"hello\"}");

    /* ── Process test commands ── */
    while (1) {
        char *raw = test_recv_str();
        if (!raw) {
            ESP_LOGW(TAG, "No more commands — disconnected or done");
            break;
        }

        char *cmd = json_get_cmd(raw);
        if (!cmd) {
            ESP_LOGW(TAG, "Invalid command JSON");
            free(raw);
            continue;
        }

        ESP_LOGI(TAG, "Command: %s", cmd);

        if (strcmp(cmd, "t1_ok") == 0) {
            handle_t1_ok();
        } else if (strcmp(cmd, "t2_capture") == 0) {
            handle_t2_capture();
        } else if (strcmp(cmd, "t3_spi_send") == 0) {
            handle_t3_spi_send();
        } else if (strcmp(cmd, "t5_full_cycle") == 0) {
            handle_t5_full_cycle();
        } else if (strcmp(cmd, "done") == 0) {
            ESP_LOGI(TAG, "Test suite complete!");
            free(raw);
            break;
        } else {
            ESP_LOGW(TAG, "Unknown command: %s", cmd);
        }

        free(raw);
    }

    close(s_test_sock);
    s_test_sock = -1;

    ESP_LOGI(TAG, "Camera test mode finished.");
    return ESP_OK;
}
