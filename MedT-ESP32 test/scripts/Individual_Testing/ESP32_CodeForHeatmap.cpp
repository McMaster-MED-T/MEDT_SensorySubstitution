
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Arduino.h>
#include <ArduinoJson.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

const int num_rows = 4;
const int num_cols = 4;
const int rowPins[num_rows] = {23, 19, 18, 5};
const int colChannels[num_cols] = {8, 9, 10, 11};

int num_LED = num_rows*num_cols;

int depthGrid[16] = {0};

char inputBuffer[128];
int inputIndex = 0;

bool discardInputLine = false;  // Discard an oversized line through its newline.
const int depthIntensity[3] = {25, 60, 100};  // PWM intensity levels.
const uint32_t blinkHalfPeriodMs[3] = {500, 250, 125};  // 1, 2, 4 Hz respectively.
uint8_t brightnessGrid[16] = {0};

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(5);

  Wire.begin();
  pwm.begin();
  pwm.setPWMFreq(1000);

  for (int r = 0; r < 4; r++) {
    pinMode(rowPins[r], OUTPUT);
    digitalWrite(rowPins[r], LOW);
  }

  // All columns OFF (active-low LED convention).
  for (int c = 0; c < 4; c++) {
    pwm.setPin(colChannels[c], 4096);
  }
}

void receiveDepthData() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n') {
      inputBuffer[inputIndex] = '\0';

      if (!discardInputLine && inputIndex > 0) {
#if ARDUINOJSON_VERSION_MAJOR == 7
        JsonDocument doc;
#else
        StaticJsonDocument<JSON_OBJECT_SIZE(2) + JSON_ARRAY_SIZE(16) + 64> doc;
#endif
        bool valid = !deserializeJson(doc, inputBuffer);
        JsonArrayConst states = doc["states"].as<JsonArrayConst>();
        valid = valid && doc["frame_id"].is<uint32_t>() && states.size() == 16;

        int newValues[16];
        for (int i = 0; valid && i < num_LED; i++) {
          if (!states[i].is<uint8_t>()) {
            valid = false;
            break;
          }
          uint8_t state = states[i].as<uint8_t>();
          if ((state >> 2) > 2 || (state & 3) > 2) {
            valid = false;
            break;
          }
          newValues[i] = state;
        }

        if (valid) {
          for (int i = 0; i < num_LED; i++) {
            brightnessGrid[i] = newValues[i] >> 2;
            depthGrid[i] = depthIntensity[newValues[i] & 3];
          }
        }
      }

      inputIndex = 0;
      discardInputLine = false;
    }
    else if (!discardInputLine) {
      if (ch == '\0' || inputIndex >= (int)sizeof(inputBuffer) - 1) {
        discardInputLine = true;  // Discard through LF, including overflow tails.
        inputIndex = 0;
      } else {
        inputBuffer[inputIndex++] = ch;  // Accept CRLF as JSON whitespace.
      }
    }
  }
}

void setColumnBrightness(int channel, int percent) {
  percent = constrain(percent, 0, 100);

  // Convert 0–100% to 0–4096 PWM counts.
  int brightness = (percent * 4096L) / 100;

  // Your LEDs are active-low:
  // setPin(channel, 0)    = ON
  // setPin(channel, 4096) = OFF
  //
  // Invert the PWM duty so a larger percentage
  // produces a brighter LED.
  if (brightness >= 4096) {
    pwm.setPin(channel, 0);       // Fully ON
  }
  else if (brightness <= 0) {
    pwm.setPin(channel, 4096);   // Fully OFF
  }
  else {
    pwm.setPin(channel, 4096 - brightness);
  }
}

void loop() {
  receiveDepthData();

  // Multiplex the four rows.
  for (int row = 0; row < num_rows; row++) {
    // Turn off all rows before changing columns.
    int flag_turn_on_row = 0;
    for (int r = 0; r < num_rows; r++) {
      
      digitalWrite(rowPins[r], LOW);
    }

    // Each row's four percentages control its LEDs.
    for (int col = 0; col < num_cols; col++) {
      //transposes the heat maps' output:
      int ledIndex = row + 4*col;
      bool blinkOn = (millis() / blinkHalfPeriodMs[brightnessGrid[ledIndex]]) % 2 == 0;
      int visiblePercent = blinkOn ? depthGrid[ledIndex] : 0;
      setColumnBrightness(
        colChannels[col],
        visiblePercent
      );
      if(visiblePercent>0){flag_turn_on_row = 1;}
    }

    // Activate the selected row.
    if(flag_turn_on_row){
    digitalWrite(rowPins[row], HIGH);
    }
    delayMicroseconds(1000);
    digitalWrite(rowPins[row], LOW);

    receiveDepthData();
  }
}
