#include <Wire.h>
#include <Adafruit_MLX90640.h>
#include <string.h>

// After exporting from Edge Impulse as Arduino library, include your project header.
// Example: #include <your_project_inferencing.h>
#include <ai_thermal_imaging_2_inferencing.h>
#define HAVE_EI 1

Adafruit_MLX90640 mlx;
float frame[32 * 24];

// Adjust these to match dataset normalization used in export_edge_impulse_images.py
static constexpr float kMinTemp = 10.0f;
static constexpr float kMaxTemp = 80.0f;
static constexpr float kDetectThreshold = 0.26f;
static constexpr float kDetectThresholdOn = 0.24f;
static constexpr float kDetectThresholdOff = 0.14f;
static constexpr float kConfidenceEmaAlpha = 0.45f;
static constexpr uint8_t kMinConsecutiveOnFrames = 1;
static constexpr uint8_t kMinConsecutiveOffFrames = 3;
static constexpr unsigned long kAlertCooldownMs = 3000;
static constexpr float kFallbackHotspotTemp = 34.0f;
static constexpr float kHybridHotspotMinTemp = 30.0f;
static constexpr float kHybridHotspotMinDelta = 6.0f;
static constexpr float kHybridHotspotMaxTempSpan = 6.0f;
static constexpr float kHotRegionThresholdBelowPeak = 2.5f;
static constexpr int kHotRegionMinPixels = 8;
static constexpr int kHotRegionMinWidth = 2;
static constexpr int kHotRegionMinHeight = 3;
static constexpr float kHotRegionMinHeightToWidth = 0.65f;
static constexpr bool kEmitRawFrames = true;
static constexpr uint8_t kFrameStreamEveryNLoops = 2;

static unsigned long g_last_alert_ms = 0;
static unsigned long g_loop_count = 0;
static bool g_detected_state = false;
static bool g_confidence_ema_initialized = false;
static float g_confidence_ema = 0.0f;
static uint8_t g_on_streak = 0;
static uint8_t g_off_streak = 0;

struct DetectionEvent {
  bool detected;
  float confidence;
  float confidence_raw;
  float x_norm;
  float y_norm;
  float w_norm;
  float h_norm;
  float min_temp;
  float max_temp;
  const char *label;
};

static float normalize_temp(float v) {
  if (v < kMinTemp) v = kMinTemp;
  if (v > kMaxTemp) v = kMaxTemp;
  return (v - kMinTemp) / (kMaxTemp - kMinTemp);
}

static bool is_person_label(const char *label) {
  if (!label) {
    return false;
  }
  return strstr(label, "person") != nullptr || strstr(label, "human") != nullptr;
}

static void summarize_frame(const float *src, float &min_temp, float &max_temp) {
  min_temp = src[0];
  max_temp = src[0];
  for (int i = 1; i < 32 * 24; i++) {
    if (src[i] < min_temp) min_temp = src[i];
    if (src[i] > max_temp) max_temp = src[i];
  }
}

static void find_hottest_pixel(const float *src, int &best_idx, float &best_temp) {
  best_idx = 0;
  best_temp = src[0];
  for (int i = 1; i < 32 * 24; i++) {
    if (src[i] > best_temp) {
      best_temp = src[i];
      best_idx = i;
    }
  }
}

static bool estimate_hot_region(const float *src,
                                float peak_temp,
                                int &min_x,
                                int &max_x,
                                int &min_y,
                                int &max_y,
                                int &hot_pixels) {
  float threshold = peak_temp - kHotRegionThresholdBelowPeak;
  min_x = 31;
  max_x = 0;
  min_y = 23;
  max_y = 0;
  hot_pixels = 0;

  for (int y = 0; y < 24; y++) {
    for (int x = 0; x < 32; x++) {
      float v = src[y * 32 + x];
      if (v < threshold) {
        continue;
      }
      hot_pixels++;
      if (x < min_x) min_x = x;
      if (x > max_x) max_x = x;
      if (y < min_y) min_y = y;
      if (y > max_y) max_y = y;
    }
  }

  return hot_pixels > 0;
}

static void apply_temporal_filter(DetectionEvent &ev) {
  ev.confidence_raw = ev.confidence;
  float conf = ev.confidence;
  if (conf < 0.0f) conf = 0.0f;
  if (conf > 1.0f) conf = 1.0f;

  if (!g_confidence_ema_initialized) {
    g_confidence_ema = conf;
    g_confidence_ema_initialized = true;
  } else {
    g_confidence_ema = (kConfidenceEmaAlpha * conf) + ((1.0f - kConfidenceEmaAlpha) * g_confidence_ema);
  }

  if (!g_detected_state) {
    if (g_confidence_ema >= kDetectThresholdOn) {
      if (g_on_streak < 255) g_on_streak++;
      if (g_on_streak >= kMinConsecutiveOnFrames) {
        g_detected_state = true;
      }
    } else {
      g_on_streak = 0;
    }
    g_off_streak = 0;
  } else {
    if (g_confidence_ema <= kDetectThresholdOff) {
      if (g_off_streak < 255) g_off_streak++;
      if (g_off_streak >= kMinConsecutiveOffFrames) {
        g_detected_state = false;
      }
    } else {
      g_off_streak = 0;
    }
    g_on_streak = 0;
  }

  ev.detected = g_detected_state;
  ev.confidence = g_confidence_ema;
  if (!ev.detected) {
    ev.label = "none";
    ev.x_norm = 0.0f;
    ev.y_norm = 0.0f;
    ev.w_norm = 0.0f;
    ev.h_norm = 0.0f;
  }
}

static void emit_event_json(const DetectionEvent &ev, bool alert_sent, const char *mode) {
  Serial.print("{\"ts_ms\":");
  Serial.print(millis());
  Serial.print(",\"mode\":\"");
  Serial.print(mode);
  Serial.print("\",\"label\":\"");
  Serial.print(ev.label);
  Serial.print("\",\"person_detected\":");
  Serial.print(ev.detected ? 1 : 0);
  Serial.print(",\"confidence\":");
  Serial.print(ev.confidence, 4);
  Serial.print(",\"confidence_raw\":");
  Serial.print(ev.confidence_raw, 4);
  Serial.print(",\"x_norm\":");
  Serial.print(ev.x_norm, 4);
  Serial.print(",\"y_norm\":");
  Serial.print(ev.y_norm, 4);
  Serial.print(",\"w_norm\":");
  Serial.print(ev.w_norm, 4);
  Serial.print(",\"h_norm\":");
  Serial.print(ev.h_norm, 4);
  Serial.print(",\"min_temp\":");
  Serial.print(ev.min_temp, 2);
  Serial.print(",\"max_temp\":");
  Serial.print(ev.max_temp, 2);
  Serial.print(",\"alert_sent\":");
  Serial.print(alert_sent ? 1 : 0);
  Serial.println("}");
}

static void emit_frame_csv(const float *src) {
  for (int i = 0; i < 32 * 24; i++) {
    if (i > 0) {
      Serial.print(',');
    }
    Serial.print(src[i], 2);
  }
  Serial.println();
}

#if HAVE_EI
static float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
static bool g_ei_ready = false;

static bool call_ei_init(void (*fn)()) {
  fn();
  return true;
}

static bool call_ei_init(EI_IMPULSE_ERROR (*fn)()) {
  return fn() == EI_IMPULSE_OK;
}

static void init_ei_once() {
  bool init_ok = call_ei_init(&run_classifier_init);
  if (init_ok) {
    g_ei_ready = true;
    Serial.println("EI init OK");
    Serial.print("EI project: ");
    Serial.println(EI_CLASSIFIER_PROJECT_NAME);
    Serial.print("EI model input: ");
    Serial.print(EI_CLASSIFIER_INPUT_WIDTH);
    Serial.print("x");
    Serial.println(EI_CLASSIFIER_INPUT_HEIGHT);
    Serial.print("EI box threshold: ");
    Serial.println(EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD, 4);
    Serial.print("EI largest arena bytes: ");
    Serial.println((unsigned long)EI_CLASSIFIER_TFLITE_LARGEST_ARENA_SIZE);
    return;
  }

  g_ei_ready = false;
  Serial.println("EI init failed");
}

static int fill_feature_data(size_t offset, size_t length, float *out_ptr) {
  memcpy(out_ptr, features + offset, length * sizeof(float));
  return 0;
}

static void prepare_features_from_frame(const float *src) {
  for (size_t y = 0; y < EI_CLASSIFIER_INPUT_HEIGHT; y++) {
    size_t src_y = (y * 24) / EI_CLASSIFIER_INPUT_HEIGHT;
    if (src_y >= 24) src_y = 23;
    for (size_t x = 0; x < EI_CLASSIFIER_INPUT_WIDTH; x++) {
      size_t src_x = (x * 32) / EI_CLASSIFIER_INPUT_WIDTH;
      if (src_x >= 32) src_x = 31;
      features[y * EI_CLASSIFIER_INPUT_WIDTH + x] = normalize_temp(src[src_y * 32 + src_x]);
    }
  }
}

static DetectionEvent run_ei_inference(const float *src) {
  DetectionEvent ev = {false, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, "none"};
  summarize_frame(src, ev.min_temp, ev.max_temp);

  if (!g_ei_ready) {
    ev.label = "ei_not_ready";
    return ev;
  }

  prepare_features_from_frame(src);

  signal_t signal;
  signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
  signal.get_data = &fill_feature_data;

  ei_impulse_result_t result = {0};
  EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
  if (err != EI_IMPULSE_OK) {
    g_ei_ready = false;
    ev.label = "infer_error";
    return ev;
  }

#if EI_CLASSIFIER_OBJECT_DETECTION == 1
  float best_person_score = 0.0f;
  float best_x = 0.0f;
  float best_y = 0.0f;
  float best_w = 0.0f;
  float best_h = 0.0f;
  for (size_t i = 0; i < EI_CLASSIFIER_OBJECT_DETECTION_COUNT; i++) {
    ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
    if (bb.value <= 0.0f || !is_person_label(bb.label)) {
      continue;
    }
    if (bb.value > best_person_score) {
      best_person_score = bb.value;
      best_x = ((float)bb.x + ((float)bb.width / 2.0f)) / (float)EI_CLASSIFIER_INPUT_WIDTH;
      best_y = ((float)bb.y + ((float)bb.height / 2.0f)) / (float)EI_CLASSIFIER_INPUT_HEIGHT;
      best_w = (float)bb.width / (float)EI_CLASSIFIER_INPUT_WIDTH;
      best_h = (float)bb.height / (float)EI_CLASSIFIER_INPUT_HEIGHT;
    }
  }
  ev.detected = best_person_score > 0.0f;
  ev.confidence = best_person_score;
  ev.x_norm = best_x;
  ev.y_norm = best_y;
  ev.w_norm = best_w;
  ev.h_norm = best_h;
  ev.label = ev.detected ? "person" : "none";
#else
  float best_person_score = 0.0f;
  const char *best_label = "none";
  for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    const char *label = ei_classifier_inferencing_categories[i];
    if (!is_person_label(label)) {
      continue;
    }
    float score = result.classification[i].value;
    if (score > best_person_score) {
      best_person_score = score;
      best_label = label;
    }
  }
  ev.detected = best_person_score > 0.0f;
  ev.confidence = best_person_score;
  ev.label = ev.detected ? best_label : "none";
#endif

  // Demo reliability fallback: if EI returns no person box, use thermal hotspot heuristic.
  if (ev.confidence <= 0.0f) {
    float delta = ev.max_temp - ev.min_temp;
    if (ev.max_temp >= kHybridHotspotMinTemp && delta >= kHybridHotspotMinDelta) {
      int hot_idx = 0;
      float hot_temp = ev.max_temp;
      find_hottest_pixel(src, hot_idx, hot_temp);

      int min_x = 0;
      int max_x = 0;
      int min_y = 0;
      int max_y = 0;
      int hot_pixels = 0;
      bool have_region = estimate_hot_region(src, hot_temp, min_x, max_x, min_y, max_y, hot_pixels);
      if (!have_region) {
        return ev;
      }

      int region_w = (max_x - min_x) + 1;
      int region_h = (max_y - min_y) + 1;
      float h_to_w = (float)region_h / (float)region_w;

      // Reject compact and flat hotspots (e.g., phone/screen corners) for fallback detection.
      if (hot_pixels < kHotRegionMinPixels ||
          region_w < kHotRegionMinWidth ||
          region_h < kHotRegionMinHeight ||
          h_to_w < kHotRegionMinHeightToWidth) {
        return ev;
      }

      int hot_y = hot_idx / 32;
      int hot_x = hot_idx % 32;
      float conf = (hot_temp - kHybridHotspotMinTemp) / kHybridHotspotMaxTempSpan;
      if (conf < 0.0f) conf = 0.0f;
      if (conf > 1.0f) conf = 1.0f;

      ev.confidence = conf;
      ev.x_norm = ((float)hot_x + 0.5f) / 32.0f;
      ev.y_norm = ((float)hot_y + 0.5f) / 24.0f;
      ev.w_norm = 0.35f;
      ev.h_norm = 0.45f;
      ev.label = "person_hotspot";
      ev.detected = ev.confidence >= kDetectThreshold;
    }
  }

  return ev;
}
#else
static DetectionEvent run_fallback_inference(const float *src) {
  DetectionEvent ev = {false, 0.0f, 0.0f, 0.5f, 0.5f, 0.4f, 0.4f, 0.0f, 0.0f, "hotspot"};
  summarize_frame(src, ev.min_temp, ev.max_temp);
  if (ev.max_temp >= kFallbackHotspotTemp) {
    float conf = (ev.max_temp - kFallbackHotspotTemp) / 8.0f;
    if (conf < 0.0f) conf = 0.0f;
    if (conf > 1.0f) conf = 1.0f;
    ev.detected = conf >= kDetectThreshold;
    ev.confidence = conf;
  }
  return ev;
}
#endif

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  Wire.begin();
  Wire.setClock(100000);

  if (!mlx.begin(MLX90640_I2CADDR_DEFAULT, &Wire)) {
    Serial.println("ERROR: MLX90640 not found");
    while (1) {
      delay(1000);
    }
  }

  mlx.setMode(MLX90640_CHESS);
  mlx.setResolution(MLX90640_ADC_18BIT);
  mlx.setRefreshRate(MLX90640_2_HZ);

#if HAVE_EI
  init_ei_once();
#endif

  Serial.println("READY: EI telemetry inference");
}

void loop() {
  g_loop_count++;

  if (mlx.getFrame(frame) != 0) {
    Serial.print("{\"ts_ms\":");
    Serial.print(millis());
    Serial.println(",\"mode\":\"error\",\"frame_error\":1}");
    delay(20);
    return;
  }

  DetectionEvent ev;
#if HAVE_EI
  ev = run_ei_inference(frame);
  apply_temporal_filter(ev);
  const char *mode = "edge_impulse";
#else
  ev = run_fallback_inference(frame);
  const char *mode = "fallback";
#endif

  bool should_alert = ev.detected && ev.confidence >= kDetectThreshold;
  bool alert_sent = false;
  if (should_alert && (millis() - g_last_alert_ms) >= kAlertCooldownMs) {
    g_last_alert_ms = millis();
    alert_sent = true;
  }

  emit_event_json(ev, alert_sent, mode);

  if (kEmitRawFrames && (g_loop_count % kFrameStreamEveryNLoops == 0)) {
    emit_frame_csv(frame);
  }

  delay(100);
}
