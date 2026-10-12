/**
 * @file wifi_stream.c
 * @brief WiFi STA mode and TCP server — IMG4 protocol for free exploration.
 */
#include "wifi_stream.h"
#include "config.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <errno.h>

static const char *TAG = "wifi_stream";

#ifndef CONFIG_WIFI_SSID
#define CONFIG_WIFI_SSID     "Network"
#endif
#ifndef CONFIG_WIFI_PASSWORD
#define CONFIG_WIFI_PASSWORD "Excels!or"
#endif

static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT  BIT0
#define WIFI_FAIL_BIT       BIT1

static int s_server_sock = -1;
static int s_client_sock = -1;
static int s_retry_count = 0;
#define MAX_RETRY 10

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_count < MAX_RETRY) {
            esp_wifi_connect();
            s_retry_count++;
            ESP_LOGW(TAG, "Retrying WiFi (%d/%d)", s_retry_count, MAX_RETRY);
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Connected! IP: " IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_count = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

esp_err_t wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id, instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, &instance_got_ip));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid     = CONFIG_WIFI_SSID,
            .password = CONFIG_WIFI_PASSWORD,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Waiting for WiFi (SSID: %s)...", CONFIG_WIFI_SSID);

    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT, pdFALSE, pdFALSE,
        pdMS_TO_TICKS(30000));

    if (bits & WIFI_CONNECTED_BIT) {
        /* Disable power saving — keep radio awake for stable streaming */
        esp_wifi_set_ps(WIFI_PS_NONE);
        ESP_LOGI(TAG, "WiFi power save disabled for streaming");
        return ESP_OK;
    }
    ESP_LOGE(TAG, "WiFi connection failed");
    return ESP_FAIL;
}

esp_err_t stream_server_start(uint16_t port)
{
    struct sockaddr_in server_addr = {
        .sin_family      = AF_INET,
        .sin_addr.s_addr = htonl(INADDR_ANY),
        .sin_port        = htons(port),
    };

    s_server_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s_server_sock < 0) {
        ESP_LOGE(TAG, "Socket failed: %d", errno);
        return ESP_FAIL;
    }

    int opt = 1;
    setsockopt(s_server_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    if (bind(s_server_sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        ESP_LOGE(TAG, "Bind failed: %d", errno);
        close(s_server_sock);
        return ESP_FAIL;
    }

    if (listen(s_server_sock, 1) < 0) {
        ESP_LOGE(TAG, "Listen failed: %d", errno);
        close(s_server_sock);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "TCP server listening on port %d (non-blocking accept)", port);
    return ESP_OK;
}

/**
 * @brief Background task that accepts TCP clients.
 *
 * Runs in a loop: wait for client → serve until disconnect → wait again.
 * The collection/navigation loop is NOT blocked by this.
 */
static void stream_accept_task(void *pvParam)
{
    while (1) {
        ESP_LOGI(TAG, "Waiting for TCP client...");

        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int client = accept(s_server_sock,
                            (struct sockaddr *)&client_addr, &addr_len);
        if (client < 0) {
            ESP_LOGW(TAG, "Accept failed: %d, retrying...", errno);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        /* Configure socket for reliability */
        /* Send timeout — prevents send() from blocking forever */
        struct timeval snd_tv = { .tv_sec = 5, .tv_usec = 0 };
        setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &snd_tv, sizeof(snd_tv));

        /* TCP_NODELAY — disable Nagle, send packets immediately */
        int flag = 1;
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        /* TCP keepalive — detect dead connections in ~25s */
        setsockopt(client, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));
        int idle = 10;   /* start probes after 10s idle */
        int intvl = 5;   /* probe every 5s */
        int cnt = 3;     /* 3 failed probes = dead */
        setsockopt(client, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
        setsockopt(client, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
        setsockopt(client, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof(cnt));

        s_client_sock = client;
        ESP_LOGI(TAG, "Client connected! (snd_timeout=5s, keepalive=10/5/3)");

        /* Wait until client disconnects (s_client_sock set to -1 by send_all) */
        while (s_client_sock >= 0) {
            vTaskDelay(pdMS_TO_TICKS(500));
        }

        ESP_LOGI(TAG, "Client disconnected, will accept new connection");
    }
}

void stream_accept_start(void)
{
    xTaskCreatePinnedToCore(
        stream_accept_task, "tcp_accept",
        4096, NULL, 3, NULL, 0
    );
}

static esp_err_t send_all(const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t *)buf;
    size_t offset = 0;
    while (offset < len) {
        int sent = send(s_client_sock, p + offset, len - offset, 0);
        if (sent < 0) {
            ESP_LOGW(TAG, "Client disconnected (err=%d)", errno);
            close(s_client_sock);
            s_client_sock = -1;
            return ESP_FAIL;
        }
        offset += sent;
    }
    return ESP_OK;
}

esp_err_t stream_send_frame(uint32_t frame_id, uint32_t timestep,
                            uint8_t action_taken, uint8_t depth_blocked,
                            float heading_deg,
                            const uint8_t *jpeg_buf, uint32_t jpeg_len)
{
    if (s_client_sock < 0) return ESP_FAIL;

    /*
     * IMG4 packet layout (22-byte header):
     *   [0..3]   magic        0x494D4734
     *   [4..7]   frame_id     uint32 LE
     *   [8..11]  timestep     uint32 LE
     *   [12]     action_taken uint8
     *   [13]     depth_blocked uint8
     *   [14..17] heading_deg  float32 LE
     *   [18..21] jpeg_len     uint32 LE
     *   [22..N]  jpeg_data
     */
    uint8_t header[22];
    uint32_t magic = STREAM_MAGIC;
    memcpy(&header[0],  &magic, 4);
    memcpy(&header[4],  &frame_id, 4);
    memcpy(&header[8],  &timestep, 4);
    header[12] = action_taken;
    header[13] = depth_blocked;
    memcpy(&header[14], &heading_deg, 4);
    memcpy(&header[18], &jpeg_len, 4);

    if (send_all(header, sizeof(header)) != ESP_OK) return ESP_FAIL;
    return send_all(jpeg_buf, jpeg_len);
}

bool stream_is_connected(void)
{
    return (s_client_sock >= 0);
}

