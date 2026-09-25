/*
 * =========================================================================
 * SAKHA KWS — ESP32-S3 Integrated Firmware (SIH Final)
 * =========================================================================
 * Features:
 *   - KWS (DS-CNN int8, N_FFT=512, N_MFCC=13) on Core 1
 *   - WiFi + WebSocket audio streaming on Core 0 (arduinoWebSockets)
 *   - Boot-time calibration (5 s) — auto AGC + silence threshold
 *   - Post-detection streaming only; the Sakha keyword is never transmitted
 *   - Short-RMS VAD (200 ms, 10-hop ring) for stream-end — NOT 1 s RMS
 *     (1 s RMS smears over mid-speech pauses -> false stream-ends)
 *   - FreeRTOS queue (20 items x 640 B = 400 ms headroom), drops oldest
 *     on overflow so fresh audio always gets through inference spikes
 *   - Detailed telemetry: cpu_instant_pct, cpu_avg_pct, infer_ms, feat_ms
 *   - Auto WS reconnect; STREAMING -> LISTENING fallback on disconnect
 *
 * Boot sequence:
 *   WiFi connect (blocking, 10 s) -> OLED "WiFi OK"
 *   -> Calibration (5 s, stay quiet) -> wsTask connect WebSocket
 *   -> LISTENING (KWS active)
 *
 * Wiring:
 *   INMP441:  SCK -> GPIO 41   WS -> GPIO 40   SD -> GPIO 42   L/R -> GND
 *   OLED SSD1306: SDA -> GPIO 8   SCL -> GPIO 9
 *
 * Required sketch files (same folder as this .ino):
 *   mel_sparse.h, dct_matrix.h, hann_window.h, model_data.h,
 *   kiss_fft.h/.c, kiss_fftr.h/.c, _kiss_fft_guts.h, kiss_fft_log.h
 *
 * Required Arduino libraries (Library Manager):
 *   - TFLiteMicro_ArduinoESP32S3
 *   - Adafruit SSD1306 + Adafruit GFX Library
 *   - arduinoWebSockets by Markus Sattler (Links2004)
 * =========================================================================
 */

// =========================================================================
// INCLUDES
// =========================================================================
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <driver/i2s.h>
#include <esp_heap_caps.h>
#include "kiss_fftr.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"   // g_model[], g_model_len
#include "dct_matrix.h"   // DCT_MATRIX[13][40]
#include "hann_window.h"  // HANN_WINDOW[512]
#include "mel_sparse.h"   // MEL_START, MEL_LEN, MEL_VALUES

// =========================================================================
// CONFIG — Edit before flashing
// =========================================================================

// Keep real values in the ignored secrets.h file. The fallback placeholders
// make a clean GitHub clone compile without exposing network credentials.
#if __has_include("secrets.h")
#include "secrets.h"
#else
#define WIFI_SSID "YOUR_WIFI_SSID"
#define WIFI_PASS "YOUR_WIFI_PASSWORD"
#define WS_HOST "YOUR_BACKEND_IP"
#endif

// ---- WiFi ---------------------------------------------------------------
#define WIFI_TIMEOUT_MS 10000

// ---- WebSocket backend --------------------------------------------------
#define WS_PORT 8000
#define WS_PATH "/ws/esp32"

// ---- Audio pipeline -----------------------------------------------------
#define SAMPLE_RATE 16000
#define N_FFT 512
#define HOP_LENGTH 320  // 20 ms per hop at 16 kHz
#define NUM_FRAMES 49
#define NUM_MFCC 13
#define NUM_MEL_FILTERS 40
#define NUM_FFT_BINS (N_FFT / 2 + 1)          // 257
#define OVERLAP_SAMPLES (N_FFT - HOP_LENGTH)  // 192

// ---- KWS decision -------------------------------------------------------
#define NUM_CLASSES 3
static const char* CLASS_LABELS[NUM_CLASSES] = { "Keyword", "Unknown", "Silence" };
#define KW_THRESHOLD 0.78f    // tune: 0.75 easy, 0.92 strict
#define KW_STREAK_N 3         // consecutive inferences above threshold
#define KW_COOLDOWN_MS 2000   // ignore re-fires within this window
#define INFER_EVERY_N_HOPS 6  // 6 x 20 ms = 120 ms between inferences

// ---- I2S pins -----------------------------------------------------------
#define I2S_PORT I2S_NUM_0
#define I2S_BCLK_PIN 41
#define I2S_WS_PIN 40
#define I2S_SD_PIN 42

// ---- OLED ---------------------------------------------------------------
#define OLED_SDA_PIN 8
#define OLED_SCL_PIN 9
#define OLED_ADDR 0x3C
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

// ---- Boot-time calibration ----------------------------------------------
#define CALIB_DURATION_MS 5000
#define CALIB_TARGET_RMS 0.06f  // -20 dBFS target
#define CALIB_GAIN_MIN 1.0f
#define CALIB_GAIN_MAX 8.0f
#define CALIB_PERCENTILE 0.15f
#define CALIB_MAX_SAMPLES 200

// ---- Streaming / VAD ----------------------------------------------------
// VAD uses g_short_rms (200 ms window), NOT g_raw_rms (1 s window).
// g_raw_rms smears over mid-speech pauses and triggers false stream-ends.
#define STREAM_SILENCE_MS 1500  // ms of silence (g_short_rms < th) -> end_stream
#define STREAM_MAX_MS 10000     // hard max stream duration (backend limit = 10 s)

// ---- PCM queue (Core 1 -> Core 0) ---------------------------------------
// Inference blocks Core 1 ~70 ms (~3-4 hops). Queue absorbs that.
// 20 items x 20 ms = 400 ms headroom.
#define PCM_QUEUE_ITEMS 20
#define PCM_CHUNK_SAMPLES HOP_LENGTH
#define PCM_CHUNK_BYTES (PCM_CHUNK_SAMPLES * sizeof(int16_t))  // 640 bytes

// ---- TFLite -------------------------------------------------------------
constexpr int kTensorArenaSize = 96 * 1024;

// =========================================================================
// STATE MACHINE
// =========================================================================
enum AppState { LISTENING,
                DETECTED,
                STREAMING };
static volatile AppState g_state = LISTENING;

// =========================================================================
// GLOBALS
// =========================================================================

// ---- OLED ---------------------------------------------------------------
static Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ---- MFCC / FFT ---------------------------------------------------------
static float frame_buf[N_FFT];
static float mfcc_frames[NUM_FRAMES][NUM_MFCC];
static kiss_fftr_cfg fft_cfg = nullptr;
static kiss_fft_scalar fft_time_buf[N_FFT];
static kiss_fft_cpx fft_freq_buf[NUM_FFT_BINS];
static float mag_spectrum[NUM_FFT_BINS];
static float mel_energies[NUM_MEL_FILTERS];

// ---- Long RMS ring (1 s, DC-corrected) — calibration + silence gate -----
// Allocated from PSRAM in setup() to save DRAM (16 000 floats = 64 KB).
#define RMS_RING_SAMPLES 16000
static float* rms_ring = nullptr;
static int rms_write_idx = 0;
static bool rms_ring_full = false;
static float g_raw_rms = 0.0f;  // updated every 16 hops (~320 ms)

// ---- Short RMS (200 ms, per-hop mean) — VAD during streaming ------------
// Each entry is the RMS of one 320-sample hop (~20 ms).
// 10 entries x 20 ms = 200 ms window.
// Fast amplitude tracking; unaffected by mid-speech pauses.
#define SHORT_RMS_HOPS 10
static float hop_rms_ring[SHORT_RMS_HOPS];
static int hop_rms_idx = 0;
static bool hop_rms_full = false;
static float g_short_rms = 0.0f;  // updated every hop inside readAudioChunk()

// ---- WebSocket (Core 0) -------------------------------------------------
static WebSocketsClient webSocket;
static volatile bool g_ws_connected = false;
static volatile bool g_send_start_stream = false;
static volatile bool g_send_end_stream = false;
static char g_end_stream_reason[32] = "silence";

// ---- PCM queue: Core 1 -> Core 0 ----------------------------------------
static QueueHandle_t pcm_queue = nullptr;

// ---- Calibration --------------------------------------------------------
static float g_calib_gain = 1.0f;
static float g_calib_ambient_rms = 0.02f;
static float g_calib_buf[CALIB_MAX_SAMPLES];
static int g_calib_count = 0;
static uint32_t g_calib_start_ms = 0;
static volatile bool g_calib_running = false;
static volatile bool g_calib_done = false;
static float g_silence_rms_th = 0.035f;

// ---- TFLite Micro -------------------------------------------------------
static DRAM_ATTR uint8_t tensor_arena[kTensorArenaSize];
static const tflite::Model* tf_model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// ---- Confidences --------------------------------------------------------
static float g_last_confidences[NUM_CLASSES] = { 0 };

// ---- CPU / timing stats -------------------------------------------------
static float g_cpu_instant_pct = 0.0f;  // single-hop spike
static float g_cpu_avg_pct = 0.0f;      // EMA-smoothed (~20-hop window)
static uint32_t g_infer_us = 0;
static uint32_t g_feat_us = 0;

// ---- Streaming state ----------------------------------------------------
static uint32_t silence_streak_ms = 0;
static uint32_t stream_start_ms = 0;

// =========================================================================
// WebSocket event handler   (called on Core 0 inside webSocket.loop())
// =========================================================================
static void onWsEvent(WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      g_ws_connected = true;
      Serial.printf("[WS] Connected to %s:%d%s\n", WS_HOST, WS_PORT, WS_PATH);
      break;

    case WStype_DISCONNECTED:
      g_ws_connected = false;
      // Core 1 checks g_ws_connected every hop; STREAMING -> LISTENING fallback
      // happens there on the next audio cycle.
      Serial.println("[WS] Disconnected — will reconnect automatically...");
      break;

    case WStype_ERROR:
      Serial.println("[WS] Error event");
      break;

    default:
      break;
  }
}

// =========================================================================
// Telemetry JSON — sent from Core 0 every 2 s
// Includes both legacy "cpu_avg" field AND new detailed fields.
// =========================================================================
static void sendTelemetry() {
  if (!g_ws_connected) return;

  uint32_t free_heap = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  uint32_t total_heap = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
  uint32_t used_heap = total_heap - free_heap;
  uint32_t ram_pct = total_heap ? (used_heap * 100 / total_heap) : 0;
  const char* state_str =
    (g_state == LISTENING) ? "Listening" : (g_state == DETECTED) ? "Detected"
                                                                 : "Streaming";

  char json[400];
  snprintf(json, sizeof(json),
           "{\"type\":\"telemetry\","
           "\"cpu_avg\":%.1f,\"cpu_c0\":0,\"cpu_c1\":%.1f,"
           "\"cpu_instant_pct\":%.1f,\"cpu_avg_pct\":%.1f,"
           "\"infer_ms\":%.2f,\"feat_ms\":%.2f,"
           "\"ram_used_kb\":%u,\"ram_free_kb\":%u,"
           "\"ram_total_kb\":%u,\"ram_pct\":%u,"
           "\"state\":\"%s\"}",
           g_cpu_avg_pct,  // legacy backend field
           g_cpu_avg_pct,
           g_cpu_instant_pct,
           g_cpu_avg_pct,
           (float)g_infer_us / 1000.0f,
           (float)g_feat_us / 1000.0f,
           (unsigned)(used_heap / 1024),
           (unsigned)(free_heap / 1024),
           (unsigned)(total_heap / 1024),
           (unsigned)ram_pct,
           state_str);

  webSocket.sendTXT(json);
}

// =========================================================================
// wsTask — Core 0
// Responsibilities: WS lifecycle, PCM drain, telemetry.
// Does NOT connect until g_calib_done is true (calibration must finish
// first so OLED shows clean "Calibrating" without WS-handshake noise).
// =========================================================================
static void wsTask(void* param) {

  // ---- Wait for calibration to finish before opening WS ----------------
  Serial.println("[wsTask] Waiting for calibration...");
  while (!g_calib_done) vTaskDelay(pdMS_TO_TICKS(100));
  Serial.println("[wsTask] Calibration done — connecting WebSocket...");

  webSocket.begin(WS_HOST, WS_PORT, WS_PATH);
  webSocket.onEvent(onWsEvent);
  webSocket.setReconnectInterval(3000);
  webSocket.enableHeartbeat(15000, 3000, 2);

  uint32_t last_telemetry_ms = 0;

  for (;;) {
    webSocket.loop();

    // ------------------------------------------------------------------
    // Handle start_stream signal from Core 1
    // ------------------------------------------------------------------
    if (g_send_start_stream) {
      g_send_start_stream = false;

      if (g_ws_connected) {
        char json[96];
        snprintf(json, sizeof(json),
                 "{\"type\":\"start_stream\",\"t1_ms\":%lu}", (unsigned long)millis());
        webSocket.sendTXT(json);
        Serial.println("[wsTask] start_stream sent");
      }

      // Detection is complete; only audio captured after this point is sent.
      g_state = STREAMING;
    }

    // ------------------------------------------------------------------
    // Handle end_stream signal from Core 1
    // ------------------------------------------------------------------
    if (g_send_end_stream) {
      g_send_end_stream = false;

      if (g_ws_connected) {
        char json[96];
        snprintf(json, sizeof(json),
                 "{\"type\":\"end_stream\",\"reason\":\"%s\"}", g_end_stream_reason);
        webSocket.sendTXT(json);
        Serial.printf("[wsTask] end_stream sent (reason: %s)\n", g_end_stream_reason);
      }
      // Flush leftover PCM so next stream starts clean
      int16_t discard[PCM_CHUNK_SAMPLES];
      while (xQueueReceive(pcm_queue, discard, 0) == pdTRUE) {}
    }

    // ------------------------------------------------------------------
    // Drain live PCM queue -> binary WS frames.
    // Drain up to 4 chunks per loop so webSocket.loop() is not starved.
    // ------------------------------------------------------------------
    if (g_ws_connected && g_state == STREAMING) {
      int16_t chunk[PCM_CHUNK_SAMPLES];
      for (int drain = 0; drain < 4; drain++) {
        if (xQueueReceive(pcm_queue, chunk, 0) != pdTRUE) break;
        webSocket.sendBIN((uint8_t*)chunk, PCM_CHUNK_BYTES);
      }
    }

    // ------------------------------------------------------------------
    // Periodic telemetry (every 2 s)
    // ------------------------------------------------------------------
    if (millis() - last_telemetry_ms > 2000) {
      last_telemetry_ms = millis();
      sendTelemetry();
    }

    vTaskDelay(pdMS_TO_TICKS(1));  // yield to WiFi/TCP stack
  }
}

// =========================================================================
// I2S Setup
// =========================================================================
static void setupI2S() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_STAND_I2S),
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  i2s_pin_config_t pins = {
    .bck_io_num = I2S_BCLK_PIN,
    .ws_io_num = I2S_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD_PIN
  };
  esp_err_t err = i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("i2s_driver_install failed: %d\n", err);
    while (1) delay(1000);
  }
  err = i2s_set_pin(I2S_PORT, &pins);
  if (err != ESP_OK) {
    Serial.printf("i2s_set_pin failed: %d\n", err);
    while (1) delay(1000);
  }
  i2s_zero_dma_buffer(I2S_PORT);
}

// =========================================================================
// readAudioChunk — one hop (HOP_LENGTH samples)
// Updates: out[], rms_ring (long), hop_rms_ring, g_short_rms
// =========================================================================
static void readAudioChunk(float* out) {
  int32_t raw32[HOP_LENGTH];
  size_t bytes_read = 0;
  while (bytes_read < sizeof(raw32)) {
    size_t got = 0;
    i2s_read(I2S_PORT, (uint8_t*)raw32 + bytes_read,
             sizeof(raw32) - bytes_read, &got, portMAX_DELAY);
    bytes_read += got;
  }

  float hop_sq_sum = 0.0f;

  for (int i = 0; i < HOP_LENGTH; i++) {
    int32_t s24 = raw32[i] >> 8;        // 24-bit signed
    float v = (float)s24 / 8388608.0f;  // -> [-1, 1]
    out[i] = v;

    // ---- Long RMS ring (1 s) ----
    rms_ring[rms_write_idx] = v;
    if (++rms_write_idx >= RMS_RING_SAMPLES) {
      rms_write_idx = 0;
      rms_ring_full = true;
    }

    // ---- Per-hop RMS accumulation ----
    hop_sq_sum += v * v;
  }

  // ---- Short RMS: one entry per hop -> 200 ms ring ----
  float hop_rms = sqrtf(hop_sq_sum / (float)HOP_LENGTH);
  hop_rms_ring[hop_rms_idx] = hop_rms;
  if (++hop_rms_idx >= SHORT_RMS_HOPS) {
    hop_rms_idx = 0;
    hop_rms_full = true;
  }
  // g_short_rms = mean of the short ring
  {
    int n = hop_rms_full ? SHORT_RMS_HOPS : (hop_rms_idx > 0 ? hop_rms_idx : 1);
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += hop_rms_ring[i];
    g_short_rms = s / (float)n;
  }
}

// =========================================================================
// updateRawRms — DC-corrected 1-second RMS
// Expensive: iterates 16 000 floats. Call every ~16 hops (~320 ms).
// Used for: calibration sample collection + LISTENING silence gate.
// NOT used for VAD during streaming (g_short_rms handles that).
// =========================================================================
static void updateRawRms() {
  if (!rms_ring_full) {
    g_raw_rms = 0.0f;
    return;
  }
  float sum = 0.0f;
  for (int i = 0; i < RMS_RING_SAMPLES; i++) sum += rms_ring[i];
  float mean = sum / (float)RMS_RING_SAMPLES;
  float s = 0.0f;
  for (int i = 0; i < RMS_RING_SAMPLES; i++) {
    float d = rms_ring[i] - mean;
    s += d * d;
  }
  g_raw_rms = sqrtf(s / (float)RMS_RING_SAMPLES);
}

// =========================================================================
// computeMFCCFrame — one frame -> NUM_MFCC coefficients
// KissFFT (N_FFT=512) + sparse mel filterbank + DCT
// =========================================================================
static void computeMFCCFrame(const float* frame, float* mfcc_out) {
  // 1) Calibrated gain + Hann window
  float gain = g_calib_gain;
  for (int i = 0; i < N_FFT; i++) {
    float s = frame[i] * gain;
    if (s > 1.0f) s = 1.0f;
    if (s < -1.0f) s = -1.0f;
    fft_time_buf[i] = s * HANN_WINDOW[i];
  }

  // 2) Real FFT (KissFFT, N_FFT=512)
  kiss_fftr(fft_cfg, fft_time_buf, fft_freq_buf);

  // 3) Power spectrum
  for (int k = 0; k < NUM_FFT_BINS; k++) {
    float re = fft_freq_buf[k].r, im = fft_freq_buf[k].i;
    mag_spectrum[k] = re * re + im * im;
  }

  // 4) Sparse mel filterbank -> 10*log10 (matches librosa power_to_db)
  for (int m = 0; m < NUM_MEL_FILTERS; m++) {
    const int st = MEL_START[m];
    const int ln = MEL_LEN[m];
    const float* mv = MEL_VALUES[m];
    float e = 0.0f;
    for (int j = 0; j < ln; j++) e += mag_spectrum[st + j] * mv[j];
    mel_energies[m] = 10.0f * log10f(fmaxf(e, 1e-10f));
  }

  // 4b) Per-frame top-dB clamp (80 dB dynamic range)
  float max_db = -1e9f;
  for (int m = 0; m < NUM_MEL_FILTERS; m++)
    if (mel_energies[m] > max_db) max_db = mel_energies[m];
  float floor_db = max_db - 80.0f;
  for (int m = 0; m < NUM_MEL_FILTERS; m++)
    if (mel_energies[m] < floor_db) mel_energies[m] = floor_db;

  // 5) DCT -> NUM_MFCC coefficients
  for (int c = 0; c < NUM_MFCC; c++) {
    float s = 0.0f;
    for (int m = 0; m < NUM_MEL_FILTERS; m++) s += DCT_MATRIX[c][m] * mel_energies[m];
    mfcc_out[c] = s;
  }
}

// =========================================================================
// TFLite Micro
// =========================================================================
static void setupModel() {
  tf_model = tflite::GetModel(g_model);
  if (tf_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("TFLite schema mismatch — re-export your model");
    while (1) delay(1000);
  }

  static tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMaxPool2D();
  resolver.AddAveragePool2D();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddRelu();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddMean();
  resolver.AddAdd();

  static tflite::MicroInterpreter static_interp(
    tf_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interp;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed — increase kTensorArenaSize");
    while (1) delay(1000);
  }
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.printf("[Model] Input dims:");
  for (int i = 0; i < input_tensor->dims->size; i++)
    Serial.printf(" %d", input_tensor->dims->data[i]);
  Serial.printf("  type=%d\n", input_tensor->type);
  Serial.printf("[Model] in  scale=%.8f  zp=%d\n",
                input_tensor->params.scale, input_tensor->params.zero_point);
  Serial.printf("[Model] out scale=%.8f  zp=%d\n",
                output_tensor->params.scale, output_tensor->params.zero_point);
}

static void fillInputTensor() {
  bool is_int8 = (input_tensor->type == kTfLiteInt8);
  float scale = is_int8 ? input_tensor->params.scale : 1.0f;
  int zp = is_int8 ? input_tensor->params.zero_point : 0;
  int idx = 0;
  for (int t = 0; t < NUM_FRAMES; t++) {
    for (int c = 0; c < NUM_MFCC; c++) {
      float v = mfcc_frames[t][c];
      if (!isfinite(v)) v = 0.0f;
      if (is_int8) {
        int32_t q = (int32_t)roundf(v / scale) + zp;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        input_tensor->data.int8[idx++] = (int8_t)q;
      } else {
        input_tensor->data.f[idx++] = v;
      }
    }
  }
}

static void readOutputConfidences(float* out_conf) {
  static float ema[NUM_CLASSES] = { 0 };
  const float alpha = 0.4f;

  bool is_int8 = (output_tensor->type == kTfLiteInt8);
  float raw[NUM_CLASSES];
  for (int i = 0; i < NUM_CLASSES; i++) {
    if (is_int8) {
      int8_t q = output_tensor->data.int8[i];
      raw[i] = (q - output_tensor->params.zero_point) * output_tensor->params.scale;
    } else {
      raw[i] = output_tensor->data.f[i];
    }
  }

  // Silence gate: only while LISTENING (uses long 1 s RMS for ambient gating).
  // During STREAMING all audio passes through; VAD uses g_short_rms instead.
  if (g_state == LISTENING && rms_ring_full && g_raw_rms < g_silence_rms_th) {
    raw[0] = 0.0f;
    raw[1] = 0.0f;
    raw[2] = 1.0f;
  }

  for (int i = 0; i < NUM_CLASSES; i++) {
    ema[i] = alpha * raw[i] + (1.0f - alpha) * ema[i];
    out_conf[i] = ema[i];
  }
}

// =========================================================================
// OLED Update
// =========================================================================
static void updateOLED(float* c, uint32_t free_heap) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  display.setCursor(0, 0);
  if (g_state == DETECTED) display.println(">>> DETECTED <<<");
  else if (g_state == STREAMING) display.println(">>> STREAMING <<<");
  else display.printf("Listening... WS:%s\n", g_ws_connected ? "OK" : "--");
  display.drawLine(0, 9, SCREEN_WIDTH, 9, SSD1306_WHITE);

  int y = 12;
  for (int i = 0; i < NUM_CLASSES; i++) {
    display.setCursor(0, y);
    display.printf("%-8s %5.1f%%", CLASS_LABELS[i], c[i] * 100.0f);
    y += 10;
  }
  display.drawLine(0, y, SCREEN_WIDTH, y, SSD1306_WHITE);
  y += 3;
  display.setCursor(0, y);
  if (g_state == DETECTED || g_state == STREAMING) {
    display.printf("T:%.1fs CPU:%.0f%%", (millis() - stream_start_ms) / 1000.0f,
                   g_cpu_instant_pct);
  } else {
    display.printf("CPUi:%.0f%% a:%.0f%%", g_cpu_instant_pct, g_cpu_avg_pct);
  }
  y += 10;
  display.setCursor(0, y);
  display.printf("RAM:%uK  srms:%.3f", (unsigned)(free_heap / 1024), g_short_rms);
  display.display();
}

// =========================================================================
// SETUP
// =========================================================================
void setup() {
  Serial.begin(115200);
  delay(300);

#ifdef ESP_NN
  Serial.println("[LIB] ESP-NN detected at compile time");
#else
  Serial.println("[LIB] WARNING: ESP-NN not detected — inference will be slow");
#endif

  // ---- OLED init ----
  Wire.begin(OLED_SDA_PIN, OLED_SCL_PIN);
  Wire.setClock(400000);
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("OLED init failed — check wiring");
    while (1) delay(1000);
  }
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Booting Sakha KWS...");
  display.display();

  // ------------------------------------------------------------------
  // 1. WiFi — connect BEFORE calibration.
  //    WS connection happens inside wsTask AFTER calibration completes,
  //    so the "Stay quiet" OLED screen is never interrupted.
  // ------------------------------------------------------------------
  display.setCursor(0, 16);
  display.printf("WiFi: %.16s\n", WIFI_SSID);
  display.display();

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t wifi_start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - wifi_start < WIFI_TIMEOUT_MS) {
    delay(200);
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[WiFi] Connected: %s\n", WiFi.localIP().toString().c_str());
    display.setCursor(0, 26);
    display.printf("IP: %s", WiFi.localIP().toString().c_str());
    display.display();
    delay(800);
  } else {
    Serial.println("[WiFi] FAILED — KWS-only offline mode");
    display.setCursor(0, 26);
    display.println("WiFi FAIL (offline)");
    display.display();
    delay(800);
  }

  // ------------------------------------------------------------------
  // 2. Allocate the long RMS buffer — prefer PSRAM to protect DRAM headroom.
  //    rms_ring:    64 KB (16 000 floats, 1 s)
  // ------------------------------------------------------------------
  rms_ring = (float*)ps_malloc(RMS_RING_SAMPLES * sizeof(float));
  if (!rms_ring) rms_ring = (float*)malloc(RMS_RING_SAMPLES * sizeof(float));
  if (!rms_ring) {
    Serial.println("FATAL: rms_ring alloc failed");
    while (1) delay(1000);
  }

  memset(rms_ring, 0, RMS_RING_SAMPLES * sizeof(float));
  memset(frame_buf, 0, sizeof(frame_buf));
  memset(mfcc_frames, 0, sizeof(mfcc_frames));
  memset(hop_rms_ring, 0, sizeof(hop_rms_ring));

  Serial.printf("[RAM] rms_ring    @ %p  (%s)\n", (void*)rms_ring,
                esp_ptr_external_ram(rms_ring) ? "PSRAM" : "DRAM");
  // ------------------------------------------------------------------
  // 3. PCM queue (Core 1 -> Core 0)
  // ------------------------------------------------------------------
  pcm_queue = xQueueCreate(PCM_QUEUE_ITEMS, PCM_CHUNK_BYTES);
  if (!pcm_queue) {
    Serial.println("FATAL: pcm_queue create failed");
    while (1) delay(1000);
  }
  Serial.printf("[Queue] %d items x %d B = %d B headroom\n",
                PCM_QUEUE_ITEMS, (int)PCM_CHUNK_BYTES, PCM_QUEUE_ITEMS * (int)PCM_CHUNK_BYTES);

  // ------------------------------------------------------------------
  // 4. KissFFT
  // ------------------------------------------------------------------
  fft_cfg = kiss_fftr_alloc(N_FFT, 0, nullptr, nullptr);
  if (!fft_cfg) {
    Serial.println("FATAL: KissFFT alloc failed");
    while (1) delay(1000);
  }

  // ------------------------------------------------------------------
  // 5. I2S
  // ------------------------------------------------------------------
  setupI2S();

  // ------------------------------------------------------------------
  // 6. TFLite model
  // ------------------------------------------------------------------
  setupModel();

  // ------------------------------------------------------------------
  // 7. Start calibration (5 s ambient noise measurement)
  // ------------------------------------------------------------------
  g_calib_start_ms = millis();
  g_calib_running = true;
  g_calib_count = 0;
  g_calib_done = false;

  Serial.printf("\n=== CALIBRATION START (%d ms) ===\n", CALIB_DURATION_MS);
  Serial.println("Stay quiet for calibration...");

  // ------------------------------------------------------------------
  // 8. Launch wsTask on Core 0
  //    wsTask waits internally for g_calib_done before opening WS.
  //    Stack 8 KB covers WS headers + JSON snprintf + queue ops.
  // ------------------------------------------------------------------
  xTaskCreatePinnedToCore(
    wsTask,    // function
    "wsTask",  // name
    8192,      // stack bytes
    NULL,      // param
    1,         // priority
    NULL,      // handle
    0          // Core 0
  );
}

// =========================================================================
// LOOP — Core 1 (Arduino default — audio-critical)
// =========================================================================
void loop() {
  static int hop_counter = 0;
  static float peak_level = 0.0f;
  static int rms_update_ctr = 0;

  hop_counter++;

  uint32_t t_start = micros();

  // ===================================================================
  // 1) Read one hop of audio
  //    I2S blocks here ~20 ms waiting for real audio. That is correct;
  //    it is NOT burning CPU — just waiting for DMA to fill.
  // ===================================================================
  float new_samples[HOP_LENGTH];
  readAudioChunk(new_samples);
  uint32_t t_after_read = micros();

  // Peak sample (debug)
  for (int i = 0; i < HOP_LENGTH; i++) {
    float a = fabsf(new_samples[i]);
    if (a > peak_level) peak_level = a;
  }

  // Slide N_FFT-deep raw frame buffer, append new hop
  memmove(frame_buf, frame_buf + HOP_LENGTH, OVERLAP_SAMPLES * sizeof(float));
  memcpy(frame_buf + OVERLAP_SAMPLES, new_samples, HOP_LENGTH * sizeof(float));

  // ===================================================================
  // 2) MFCC: one new frame per hop, slide into 49-deep ring
  // ===================================================================
  float new_mfcc[NUM_MFCC];
  computeMFCCFrame(frame_buf, new_mfcc);
  memmove(mfcc_frames[0], mfcc_frames[1],
          (NUM_FRAMES - 1) * NUM_MFCC * sizeof(float));
  memcpy(mfcc_frames[NUM_FRAMES - 1], new_mfcc, NUM_MFCC * sizeof(float));

  // Long RMS every ~16 hops to amortize cost of iterating 16 000 floats
  if (++rms_update_ctr >= 16) {
    rms_update_ctr = 0;
    updateRawRms();
  }

  // ===================================================================
  // CALIBRATION PHASE (first 5 s after boot)
  // ===================================================================
  if (g_calib_running) {
    if (g_raw_rms > 0.0005f && g_calib_count < CALIB_MAX_SAMPLES)
      g_calib_buf[g_calib_count++] = g_raw_rms;

    // OLED every 300 ms
    static uint32_t last_oled_ms = 0;
    if (millis() - last_oled_ms > 300) {
      last_oled_ms = millis();
      display.clearDisplay();
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(0, 0);
      display.println("  CALIBRATING...");
      display.drawLine(0, 9, SCREEN_WIDTH, 9, SSD1306_WHITE);
      display.setCursor(0, 14);
      display.println("  Stay quiet...");
      int pct = (int)((millis() - g_calib_start_ms) * 100 / CALIB_DURATION_MS);
      if (pct > 100) pct = 100;
      display.setCursor(0, 28);
      display.printf("  Progress : %d%%", pct);
      display.setCursor(0, 40);
      display.printf("  Samples  : %d", g_calib_count);
      display.setCursor(0, 52);
      display.printf("  rms      : %.5f", g_raw_rms);
      display.display();
    }

    if (millis() - g_calib_start_ms > CALIB_DURATION_MS) {
      int n = g_calib_count;
      if (n > 0) {
        // Insertion sort (n <= 200 — cheap enough here)
        for (int i = 1; i < n; i++) {
          float k = g_calib_buf[i];
          int j = i - 1;
          while (j >= 0 && g_calib_buf[j] > k) {
            g_calib_buf[j + 1] = g_calib_buf[j];
            j--;
          }
          g_calib_buf[j + 1] = k;
        }
        int cidx = (int)(CALIB_PERCENTILE * (n - 1));
        g_calib_ambient_rms = g_calib_buf[cidx];
        if (g_calib_ambient_rms < 1e-4f) g_calib_ambient_rms = 1e-4f;

        float g = CALIB_TARGET_RMS / g_calib_ambient_rms;
        if (g < CALIB_GAIN_MIN) g = CALIB_GAIN_MIN;
        if (g > CALIB_GAIN_MAX) g = CALIB_GAIN_MAX;
        g_calib_gain = g;

        float th = g_calib_ambient_rms * 1.5f;
        if (th < 0.02f) th = 0.02f;
        if (th > 0.08f) th = 0.08f;
        g_silence_rms_th = th;
      }

      g_calib_running = false;
      g_calib_done = true;  // signals wsTask to connect WS now

      Serial.println("\n=== CALIBRATION COMPLETE ===");
      Serial.printf("  samples     : %d\n", n);
      Serial.printf("  ambient rms : %.5f\n", g_calib_ambient_rms);
      Serial.printf("  gain        : %.2f\n", g_calib_gain);
      Serial.printf("  silence th  : %.5f\n", g_silence_rms_th);
      Serial.println("  wsTask connecting to backend...");
      Serial.println("=== LISTENING ===\n");
    }

    peak_level = 0.0f;
    return;  // Skip inference + streaming during calibration
  }
  // ===================================================================
  // END CALIBRATION
  // ===================================================================

  uint32_t t_after_feat = micros();

  // ===================================================================
  // 3) Inference — every INFER_EVERY_N_HOPS hops (6 x 20 ms = 120 ms)
  // ===================================================================
  uint32_t t_after_infer = t_after_feat;
  bool ran_inference = false;

  if (hop_counter % INFER_EVERY_N_HOPS == 0) {
    fillInputTensor();
    if (interpreter->Invoke() == kTfLiteOk) {
      readOutputConfidences(g_last_confidences);
    } else {
      Serial.println("Invoke() failed");
    }
    t_after_infer = micros();
    ran_inference = true;
  }

  // ===================================================================
  // 4) KWS: wake-word detection (LISTENING state only)
  // ===================================================================
  static int kw_streak = 0;
  static uint32_t last_wake_ms = 0;

  if (ran_inference && g_state == LISTENING) {
    if (g_last_confidences[0] > KW_THRESHOLD) {
      kw_streak++;
      if (kw_streak >= KW_STREAK_N && (millis() - last_wake_ms) > KW_COOLDOWN_MS) {

        last_wake_ms = millis();
        kw_streak = 0;
        Serial.println("*** WAKE WORD: SAKHA ***");

        g_state = DETECTED;
        stream_start_ms = millis();
        silence_streak_ms = 0;
        g_send_start_stream = true;  // signal wsTask
      }
    } else {
      kw_streak = 0;
    }
  }

  // ===================================================================
  // 5) STREAMING: push PCM to queue + VAD stream-end detection
  // ===================================================================
  if (g_state == DETECTED || g_state == STREAMING) {

    // ---- WS disconnect recovery ----
    if (!g_ws_connected) {
      Serial.println("[STREAM] WS lost — fallback to LISTENING");
      g_state = LISTENING;
      silence_streak_ms = 0;
      // No end_stream needed — connection is already gone
    } else if (g_state == STREAMING) {
      // ---- float -> int16 (calibrated gain) -> enqueue ----
      int16_t pcm_chunk[PCM_CHUNK_SAMPLES];
      for (int i = 0; i < HOP_LENGTH; i++) {
        float v = new_samples[i] * g_calib_gain;
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        pcm_chunk[i] = (int16_t)(v * 32767.0f);
      }

      // Non-blocking enqueue. If full, drop oldest to make room for latest.
      // Inference spikes (~70 ms, ~3-4 hops) fill at most 4 items;
      // queue holds 20 -> ample margin.
      if (xQueueSend(pcm_queue, pcm_chunk, 0) != pdTRUE) {
        int16_t discard[PCM_CHUNK_SAMPLES];
        xQueueReceive(pcm_queue, discard, 0);  // evict oldest
        xQueueSend(pcm_queue, pcm_chunk, 0);   // insert latest
      }

      // ---- VAD: use g_short_rms (200 ms), NOT g_raw_rms (1 s) ----
      // g_raw_rms smears over 1-second windows and triggers false endings
      // on mid-speech pauses (this is the bug we debugged for 3 days).
      // g_short_rms = mean RMS of last 10 hops (200 ms) — tracks fast changes.
      if (hop_rms_full && g_short_rms < g_silence_rms_th) {
        silence_streak_ms += 20;  // each hop ~= 20 ms
        if (silence_streak_ms >= STREAM_SILENCE_MS) {
          Serial.printf("[STREAM] Silence %u ms — ending stream\n",
                        (unsigned)silence_streak_ms);
          g_state = LISTENING;
          silence_streak_ms = 0;
          strncpy(g_end_stream_reason, "silence", sizeof(g_end_stream_reason));
          g_send_end_stream = true;
        }
      } else {
        silence_streak_ms = 0;  // speech detected — reset streak
      }

      // ---- Hard max-duration cap (backend also enforces 10 s) ----
      if ((millis() - stream_start_ms) >= STREAM_MAX_MS) {
        Serial.println("[STREAM] Max 10 s reached — ending stream");
        g_state = LISTENING;
        silence_streak_ms = 0;
        strncpy(g_end_stream_reason, "timeout", sizeof(g_end_stream_reason));
        g_send_end_stream = true;
      }
    }
  }

  // ===================================================================
  // 6) CPU / timing stats
  // ===================================================================
  uint32_t feat_us_now = t_after_feat - t_after_read;
  uint32_t infer_us_now = t_after_infer - t_after_feat;
  // Hop budget = time available before next audio chunk must be processed
  uint32_t hop_budget = (uint32_t)((uint64_t)HOP_LENGTH * 1000000ULL / SAMPLE_RATE);  // 20 000 us

  g_cpu_instant_pct = 100.0f * (float)(feat_us_now + infer_us_now) / (float)hop_budget;
  if (g_cpu_instant_pct > 999.9f) g_cpu_instant_pct = 999.9f;
  // Slow EMA for average (alpha=0.05 => ~20-hop / 400 ms smoothing)
  g_cpu_avg_pct = 0.05f * g_cpu_instant_pct + 0.95f * g_cpu_avg_pct;

  if (ran_inference) {
    g_infer_us = infer_us_now;
    g_feat_us = feat_us_now;
  }

  uint32_t free_heap = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

  // ===================================================================
  // 7) OLED + Serial (throttled: every 10 hops AND >= 500 ms)
  // ===================================================================
  static int display_ctr = 0;
  static float last_conf[NUM_CLASSES] = { 0 };
  static uint32_t last_display_ms = 0;

  display_ctr++;
  bool conf_changed = false;
  for (int i = 0; i < NUM_CLASSES; i++) {
    if (fabsf(g_last_confidences[i] - last_conf[i]) > 0.01f) {
      conf_changed = true;
      break;
    }
  }

  uint32_t now_ms = millis();
  if (display_ctr >= 10 && (conf_changed || (now_ms - last_display_ms) > 500)) {
    display_ctr = 0;
    last_display_ms = now_ms;
    memcpy(last_conf, g_last_confidences, sizeof(last_conf));
    updateOLED(g_last_confidences, free_heap);

    const char* st =
      (g_state == LISTENING) ? "LISTEN" : (g_state == DETECTED) ? "DETECT"
                                                                : "STREAM";
    Serial.printf(
      "[%s] feat=%uus infer=%uus(%s) "
      "cpu_i=%.0f%% cpu_a=%.0f%% "
      "peak=%.3f srms=%.4f lrms=%.4f "
      "ws=%s sil=%ums  ",
      st,
      (unsigned)feat_us_now, (unsigned)infer_us_now,
      ran_inference ? "ran" : "skip",
      g_cpu_instant_pct, g_cpu_avg_pct,
      peak_level, g_short_rms, g_raw_rms,
      g_ws_connected ? "OK" : "--",
      (unsigned)silence_streak_ms);
    for (int i = 0; i < NUM_CLASSES; i++)
      Serial.printf("%s=%.2f ", CLASS_LABELS[i], g_last_confidences[i]);
    Serial.println();
    peak_level = 0.0f;
  }
}
