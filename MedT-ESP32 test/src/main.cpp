#include <Arduino.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

// ---------------------------
// MATRIX WIRING
// ---------------------------
const uint8_t rowPins[4] = {23, 19, 18, 5};
const uint8_t colChannels[4] = {8, 9, 10, 11};

// Confirmed polarity
const uint8_t ROW_ACTIVE = HIGH;
const uint8_t ROW_INACTIVE = LOW;

// Confirmed PCA sink behavior
const uint16_t COL_FULL_ON = 0;
const uint16_t COL_FULL_OFF = 4095;

// ---------------------------
// PLAYBACK BUFFER
// ---------------------------
const int MAX_FRAMES = 5000;
int states[MAX_FRAMES];
int numFrames = 0;
float fps = 30.0f;
unsigned long frameDelayMs = 33;
int currentFrameIndex = 0;
unsigned long lastFrameStep = 0;
bool readyToPlay = false;

// ---------------------------
// TAKEOVER LOGIC
// ---------------------------
int activeState = 0;
int candidateState = 0;

float dutyCycle = 0.05f;
const float dutyStep = 0.005f;
const float minDuty = 0.0f;
const float maxDuty = 1.0f;

// unstable mode
bool unstableMode = false;
int lowDutyFrameCount = 0;
const float unstableThreshold = 0.05f;
const int unstableFrames = 15;

// ---------------------------
// SOFTWARE PWM
// ---------------------------
const int pwmSteps = 100;
const unsigned long pwmStepMs = 1;

unsigned long lastPwmStepTime = 0;
int pwmPhase = 0;

// ---------------------------
// HELPERS
// ---------------------------
void disableAllRows() {
  for (int i = 0; i < 4; i++) {
    digitalWrite(rowPins[i], ROW_INACTIVE);
  }
}

void clearAllColumns() {
  for (int c = 0; c < 4; c++) {
    pwm.setPWM(colChannels[c], 0, COL_FULL_OFF);
  }
}

void enableAllColumns() {
  for (int c = 0; c < 4; c++) {
    pwm.setPWM(colChannels[c], 0, COL_FULL_ON);
  }
}

void updateStateWithTakeover(int observedState) {
  if (observedState < 0 || observedState > 3) return;

  if (observedState == activeState) {
    dutyCycle += dutyStep;
    if (dutyCycle > maxDuty) dutyCycle = maxDuty;
  } else {
    candidateState = observedState;

    dutyCycle -= dutyStep;
    if (dutyCycle < minDuty) dutyCycle = minDuty;

    if (dutyCycle <= 0.0f) {
      activeState = candidateState;
      dutyCycle = 0.05f;
    }
  }

  if (dutyCycle <= unstableThreshold) {
    lowDutyFrameCount++;
  } else {
    lowDutyFrameCount = 0;
  }

  unstableMode = (lowDutyFrameCount >= unstableFrames);
}

void renderStateSoftPWM() {
  unsigned long now = millis();
  if (now - lastPwmStepTime < pwmStepMs) return;
  lastPwmStepTime = now;

  pwmPhase = (pwmPhase + 1) % pwmSteps;
  int onSteps = (int)(dutyCycle * pwmSteps + 0.5f);

  disableAllRows();
  clearAllColumns();

  if (pwmPhase < onSteps) {
    enableAllColumns();
    digitalWrite(rowPins[activeState], ROW_ACTIVE);
  }
}

void printDebugStatus(int observedState) {
  Serial.print("Observed: ");
  Serial.print(observedState);
  Serial.print(" | Active: ");
  Serial.print(activeState);
  Serial.print(" | Candidate: ");
  Serial.print(candidateState);
  Serial.print(" | Duty: ");
  Serial.print(dutyCycle, 3);
  Serial.print(" | Unstable: ");
  Serial.println(unstableMode ? "YES" : "NO");
}

void loadPayloadFromSerial(const String& msg) {
  StaticJsonDocument<32768> doc;
  DeserializationError err = deserializeJson(doc, msg);

  if (err) {
    Serial.print("JSON parse failed: ");
    Serial.println(err.c_str());
    return;
  }

  fps = doc["fps"] | 30.0f;
  int reportedFrames = doc["num_frames"] | 0;

  JsonArray arr = doc["states"].as<JsonArray>();
  int i = 0;
  for (int value : arr) {
    if (i < MAX_FRAMES) {
      states[i++] = value;
    }
  }

  numFrames = i;

  if (reportedFrames != numFrames) {
    Serial.print("Warning: reported num_frames=");
    Serial.print(reportedFrames);
    Serial.print(", loaded=");
    Serial.println(numFrames);
  }

  if (fps <= 0.0f) fps = 30.0f;
  frameDelayMs = (unsigned long)(1000.0f / fps);

  currentFrameIndex = 0;
  lastFrameStep = millis();
  readyToPlay = (numFrames > 0);

  if (readyToPlay) {
    activeState = states[0];
    candidateState = activeState;
    dutyCycle = 0.05f;
    unstableMode = false;
    lowDutyFrameCount = 0;

    Serial.println("Sequence loaded.");
    Serial.print("FPS: ");
    Serial.println(fps);
    Serial.print("Frames: ");
    Serial.println(numFrames);
    Serial.print("Frame delay (ms): ");
    Serial.println(frameDelayMs);
    Serial.print("Initial state: ");
    Serial.println(activeState);
  } else {
    Serial.println("No states loaded.");
  }
}

// ---------------------------
// SETUP
// ---------------------------
void setup() {
  Serial.begin(115200);
  delay(500);

  Wire.begin();
  pwm.begin();
  pwm.setPWMFreq(1000);

  for (int i = 0; i < 4; i++) {
    pinMode(rowPins[i], OUTPUT);
    digitalWrite(rowPins[i], ROW_INACTIVE);
  }

  clearAllColumns();
  disableAllRows();

  Serial.println("ESP32 Task D receiver ready.");
  Serial.println("Waiting for JSON payload...");

  activeState = 0;
  candidateState = 0;
  dutyCycle = 0.0f;
}

// ---------------------------
// LOOP
// ---------------------------
void loop() {
  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
    msg.trim();
    if (msg.length() > 0) {
      loadPayloadFromSerial(msg);
    }
  }

  if (readyToPlay && currentFrameIndex < numFrames) {
    unsigned long now = millis();
    if (now - lastFrameStep >= frameDelayMs) {
      int observedState = states[currentFrameIndex];

      updateStateWithTakeover(observedState);
      printDebugStatus(observedState);

      currentFrameIndex++;
      lastFrameStep = now;
    }
  }

  renderStateSoftPWM();
}