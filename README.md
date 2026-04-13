# Hand Gesture Recognition with MPU6050 + ESP32

Real-time hand gesture detection using **XGBoost** + **Triple-Stream Signal Processing** deployed on an ESP32 microcontroller with MPU6050 IMU sensor and SSD1306 OLED display.

Recognizes 3 gestures: **Clapping**, **Fist Making**, **Thumbs Up**

---

## Model Performance

### Base Model (Triple-Stream XGBoost)
Trained on [HGAG Dataset](https://data.mendeley.com/datasets/mkhn7kxjvy/1) — 3,223 windows across 3 classes.

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Clapping     | 0.95      | 0.84   | 0.89     | 1075    |
| Fist Making  | 0.92      | 0.96   | 0.94     | 1073    |
| Thumb Up     | 0.85      | 0.91   | 0.88     | 1075    |
| **Accuracy** |           |        | **0.90** | 3223    |
| Macro Avg    | 0.91      | 0.90   | 0.90     | 3223    |

---

### After Fine-Tuning (Personal Data — VAL set)
Fine-tuned on 79 custom samples collected from our own sensor.

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Clapping     | 0.87      | 1.00   | 0.93     | 27      |
| Fist Making  | 1.00      | 0.92   | 0.96     | 26      |
| Thumb Up     | 1.00      | 0.92   | 0.96     | 26      |
| **Accuracy** |           |        | **0.95** | 79      |
| Macro Avg    | 0.96      | 0.95   | 0.95     | 79      |

---

### After Fine-Tuning (Personal Data — TEST set)

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Clapping     | 0.87      | 1.00   | 0.93     | 26      |
| Fist Making  | 1.00      | 0.93   | 0.96     | 27      |
| Thumb Up     | 1.00      | 0.93   | 0.96     | 27      |
| **Accuracy** |           |        | **0.95** | 80      |
| Macro Avg    | 0.96      | 0.95   | 0.95     | 80      |

---

## Project Structure

```
gesture-recognition/
├── data/                          # Fine-tuning dataset (personal recordings)
│   ├── clapping_clean.csv
│   ├── fist_making_clean.csv
│   └── thumbs_up_clean.csv
├── notebook/
│   └── dectect-action-in-mpu-esp32.ipynb   # Training notebook (Kaggle)
├── arduino/
│   ├── sketch_jan26a_3class2.ino  # Main Arduino/ESP32 firmware
│   └── gesture_model-1.h          # Exported XGBoost model (C header)
├── models/
│   ├── gesture_model_triple.pkl           # Base model bundle
│   └── gesture_model_triple_finetuned.pkl # Fine-tuned model bundle
└── README.md
```

---

## System Architecture

### Triple-Stream Signal Processing

Each 160-sample window @ 200 Hz is processed into **3 parallel streams**, then downsampled to 100 Hz:

| Stream | Filter          | Deadband             | Purpose                          |
|--------|-----------------|----------------------|----------------------------------|
| A      | HPF @ 2.0 Hz    | Accel: 0.08, Gyro: 0.04 | Fast motion, noise suppression |
| B      | HPF @ 0.3 Hz    | None                 | Slow motion (~0.6 Hz, Thumbs Up) |
| C      | RAW (mean-norm) | None                 | Gravity/posture change           |

Each stream produces **90 features** (15 features × 6 axes), for **270 total features**.

### Feature Extraction (15 features per axis)

| Index | Feature            |
|-------|--------------------|
| 0     | Mean               |
| 1     | Std deviation      |
| 2     | Min                |
| 3     | Max                |
| 4     | Range              |
| 5     | Mean absolute value |
| 6     | Total variation    |
| 7     | 25th percentile    |
| 8     | 75th percentile    |
| 9     | RMS                |
| 10–14 | FFT bins 1–5       |

### Model

- **Algorithm**: XGBoost classifier
- **Trees**: 52 (after fine-tuning: 40 base + 12 new)
- **Max depth**: 3
- **Learning rate**: 0.08
- **Window**: 160 raw samples → 100 processed samples (50% overlap)

---

## Datasets

### 1. Base Training Dataset — HGAG
**Source**: [Hand Gesture Accelerometer and Gyroscope Dataset (HGAG-DATA)](https://data.mendeley.com/datasets/mkhn7kxjvy/1)

- Multiple subjects, multiple sessions
- 200 Hz sampling rate
- 6-axis IMU (accel + gyro)
- Used for initial XGBoost training

### 2. Fine-Tuning Dataset — Personal
Located in `data/` folder. Collected with the actual ESP32 + MPU6050 hardware.

| File                    | Gesture      | Windows |
|-------------------------|--------------|---------|
| `clapping_clean.csv`    | Clapping     | ~109    |
| `fist_making_clean.csv` | Fist Making  | ~110    |
| `thumbs_up_clean.csv`   | Thumbs Up    | ~109    |

**Columns**: `ax, ay, az, gx, gy, gz`

Fine-tuning improves accuracy on your specific hardware by adapting to individual motion patterns and sensor characteristics.

---

## Hardware Requirements

| Component   | Model          | Notes                          |
|-------------|----------------|--------------------------------|
| MCU         | ESP32          | Any variant with I2C           |
| IMU         | MPU6050        | 6-axis accelerometer + gyro    |
| Display     | SSD1306 OLED   | 128×64, I2C address `0x3C`     |
| LED         | Any LED        | Connected to GPIO 2            |

**Wiring (I2C)**:
```
ESP32 SDA → MPU6050 SDA / SSD1306 SDA
ESP32 SCL → MPU6050 SCL / SSD1306 SCL
ESP32 3.3V → VCC
ESP32 GND → GND
ESP32 GPIO2 → LED (+ 220Ω resistor → GND)
```

---

## Getting Started

### Step 1 — Train the Model (Kaggle)

1. Upload `dectect-action-in-mpu-esp32.ipynb` to [Kaggle](https://www.kaggle.com)
2. Add the HGAG dataset:  
   `https://data.mendeley.com/datasets/mkhn7kxjvy/1`
3. Add your personal fine-tuning dataset to the input files
4. Run all cells

The notebook will:
- Train the base XGBoost model on HGAG data
- Fine-tune on your personal data
- Export `gesture_model-1.h` (C header for Arduino)
- Export `FEATURE_MEAN` and `FEATURE_SCALE` arrays for normalization

---

### Step 2 — Export the Model to C Header

After running the notebook, the model is exported using `micromlgen`:

```python
from micromlgen import port
c_code = port(model, classmap={i: name for i, name in enumerate(le.classes_)})
with open('gesture_model-1.h', 'w') as f:
    f.write(c_code)
```

This generates a `gesture_model-1.h` file that can be included directly in the Arduino sketch.

---

### Step 3 — Update Scaler Arrays in Arduino Sketch

> **Critical**: The `FEATURE_MEAN` and `FEATURE_SCALE` arrays in the `.ino` file must match the scaler fitted during training.

After training, the notebook prints the arrays. Copy-paste them into the `.ino` file:

```cpp
// In sketch_jan26a_3class2.ino — replace these arrays with values from notebook output

const float FEATURE_MEAN[NUM_FEATURES] = {
    -0.00630524f, 0.84377591f, -3.89463243f, 3.61798066f, 7.51261310f, 0.39297763f,
    104.38681443f, -0.11268998f, 0.01292176f, 0.84383251f, 1.20300363f, 0.97826604f,
    // ... (270 values total — copy from notebook cell output "Export Scaler → Arduino")
};

const float FEATURE_SCALE[NUM_FEATURES] = {
    0.00790748f, 0.55259597f, 4.67544648f, 2.39368240f, 6.78191855f, 0.27229807f,
    74.15764734f, 0.18206346f, 0.08204549f, 0.55260209f, 0.89561802f, 0.89988049f,
    // ... (270 values total)
};
```

The full values are printed by this cell in the notebook:

```python
print_c_array('FEATURE_MEAN',  scaler.mean_)
print_c_array('FEATURE_SCALE', scaler.scale_)
```

---

### Step 4 — Flash to ESP32 (Arduino IDE)

**Required Libraries** (install via Library Manager):
- `Adafruit MPU6050`
- `Adafruit SSD1306`
- `Adafruit GFX Library`
- `Adafruit Unified Sensor`

**Project files needed in the same folder**:
```
sketch_jan26a_3class2/
├── sketch_jan26a_3class2.ino
└── gesture_model-1.h          ← generated in Step 2
```

**Board settings**:
- Board: `ESP32 Dev Module` (or your variant)
- Upload Speed: `115200`
- CPU Frequency: `240 MHz`

**Flash steps**:
1. Open `sketch_jan26a_3class2.ino` in Arduino IDE
2. Place `gesture_model-1.h` in the same folder
3. Select your ESP32 board and COM port
4. Click **Upload**
5. Open Serial Monitor at 115200 baud to see predictions

---

### Step 5 — Flash via Android Studio / USB Debugging (Optional)

If you use an Android companion app (e.g., for BLE/Serial visualization):

1. Install [Arduino IDE 2.x](https://www.arduino.cc/en/software) or use [PlatformIO](https://platformio.org/)
2. Enable USB debugging on your Android device
3. The ESP32 communicates over **Serial (UART)** at 115200 baud
4. On Android, you can read predictions using a USB Serial library such as:
   - [felHR85/UsbSerial](https://github.com/felHR85/UsbSerial)
   - [mik3y/usb-serial-for-android](https://github.com/mik3y/usb-serial-for-android)

Example data format from ESP32 Serial:
```
Predicted: 0 -> Clapping
Predicted: 1 -> Fist Making
Predicted: 2 -> Thumbs Up
No motion
```

In Android Studio, parse this stream with a `BufferedReader` over the serial connection:

```kotlin
// Kotlin example snippet
val data = inputStream.bufferedReader().readLine()
if (data.startsWith("Predicted:")) {
    val gesture = data.substringAfter("-> ").trim()
    runOnUiThread { textView.text = gesture }
}
```

---

## Adding a Video Demo

To showcase your project, add a demo video in the following format:

```markdown
![Hardware Setup](demo/hardware_photo.jpg)


## Configuration Parameters

All tunable parameters are at the top of the `.ino` file:

```cpp
constexpr int WINDOW_SIZE = 100;         // Processing window (samples @ 100 Hz)
constexpr int WINDOW_OVERLAP = 50;       // 50% overlap
constexpr int RAW_SAMPLE_RATE_HZ = 200;  // Sensor sampling rate

// Stream A — High-pass filter
constexpr float HPF_A_CUTOFF_HZ  = 2.0f;
constexpr float ACCEL_DEADBAND_A = 0.08f;
constexpr float GYRO_DEADBAND_A  = 0.04f;

// Stream B — Low-frequency motion
constexpr float HPF_B_CUTOFF_HZ = 0.3f;
```

---

## Inference Pipeline (On-Device)

```
MPU6050 @ 200Hz
    ↓
[Stream A] HPF 2Hz + deadband  →  resample → 100Hz window
[Stream B] HPF 0.3Hz            →  resample → 100Hz window
[Stream C] RAW - gravity mean   →  resample → 100Hz window
    ↓
Extract 15 features × 6 axes × 3 streams = 270 features
    ↓
StandardScaler (FEATURE_MEAN / FEATURE_SCALE)
    ↓
XGBoost predict (gesture_model-1.h)
    ↓
Display on OLED + Serial output + LED control
```

---

## Notes & Troubleshooting

| Issue | Solution |
|-------|----------|
| Low accuracy on new hardware | Re-collect personal data and re-run fine-tuning |
| `FEATURE_MEAN`/`FEATURE_SCALE` mismatch | Always copy from the same notebook run that exported `gesture_model-1.h` |
| Thumbs Up hard to detect | Ensure HPF_B_CUTOFF_HZ ≤ 0.3 Hz (catches ~0.6 Hz motion) |
| Display not working | Check I2C address (try `0x3D` if `0x3C` fails) |
| Sketch too large | Reduce `XGB_N_ESTIMATORS` before exporting model |
| "No motion" constantly | Lower `MIN_ACTIVE_VALUES` or check deadband settings |

---

## License

MIT License — feel free to use, modify, and redistribute.

---

## Acknowledgements

- [HGAG Dataset](https://data.mendeley.com/datasets/mkhn7kxjvy/1) by mgriffe2004
- [micromlgen](https://github.com/eloquentarduino/micromlgen) for XGBoost → C conversion
- [Adafruit Libraries](https://github.com/adafruit) for MPU6050 & SSD1306 drivers