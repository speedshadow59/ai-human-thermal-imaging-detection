#include <Wire.h>
#include <Adafruit_MLX90640.h>

Adafruit_MLX90640 mlx;
float frame[32 * 24];

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  Wire.begin();
  Wire.setClock(400000);

  if (!mlx.begin(MLX90640_I2CADDR_DEFAULT, &Wire)) {
    Serial.println("ERROR: MLX90640 not found");
    while (1) {
      delay(1000);
    }
  }

  // Lower refresh rates improve stability on some boards/wiring setups.
  mlx.setMode(MLX90640_CHESS);
  mlx.setResolution(MLX90640_ADC_18BIT);
  mlx.setRefreshRate(MLX90640_8_HZ);

  Serial.println("READY");
}

void loop() {
  if (mlx.getFrame(frame) != 0) {
    Serial.println("FRAME_ERROR");
    delay(20);
    return;
  }

  for (int i = 0; i < 32 * 24; i++) {
    Serial.print(frame[i], 2);
    if (i < (32 * 24 - 1)) {
      Serial.print(',');
    }
  }
  Serial.println();
}
