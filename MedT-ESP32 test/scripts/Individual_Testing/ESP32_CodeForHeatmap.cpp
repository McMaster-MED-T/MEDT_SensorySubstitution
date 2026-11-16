
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Arduino.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

const int num_rows = 4;
const int num_cols = 4;
const int rowPins[num_rows] = {23, 19, 18, 5};
const int colChannels[num_cols] = {8, 9, 10, 11};

const int num_LED = num_rows*num_cols;

int depthGrid[num_LED] = {0};

// Raw layout: version, frame ID (2 bytes, little-endian), patch count,
// packed patch pairs (upper nibble first), CRC-8. COBS adds one byte.
constexpr uint8_t protocolVersion = 1;
constexpr int headerSize = 4;
constexpr int rawPacketSize = headerSize + num_LED / 2 + 1;
constexpr int encodedPacketSize = rawPacketSize + 1;
static_assert(num_LED > 0 && num_LED <= 254 && num_LED % 2 == 0,
              "Packet format requires an even patch count from 2 to 254");
uint8_t inputBuffer[encodedPacketSize];
int inputIndex = 0;
bool discardInputPacket = false;
const int depthIntensity[3] = {25, 60, 100};  // PWM intensity levels.
const uint32_t blinkHalfPeriodMs[3] = {500, 250, 125};  // 1, 2, 4 Hz respectively.
uint8_t brightnessGrid[num_LED] = {0};

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

uint8_t crc8(const uint8_t* data, int length) {
  uint8_t crc = 0;
  for (int i = 0; i < length; i++) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; bit++) {
      if (crc & 0x80) {
        crc = (crc << 1) ^ 0x07;
      } else {
        crc <<= 1;
      }
    }
  }
  return crc;
}

void applyPacket() {
  if (inputIndex != encodedPacketSize) return;

  uint8_t raw[rawPacketSize];
  int readIndex = 0;
  int writeIndex = 0;
  while (readIndex < inputIndex) {
    uint8_t code = inputBuffer[readIndex++];
    if (code == 0) return;
    int runLength = code - 1;
    if (runLength > inputIndex - readIndex || runLength > rawPacketSize - writeIndex) return;
    for (int i = 0; i < runLength; i++) {
      raw[writeIndex++] = inputBuffer[readIndex++];
    }
    // Packets are shorter than 254 bytes, so a following run marks a zero.
    if (readIndex < inputIndex) {
      if (writeIndex == rawPacketSize) return;
      raw[writeIndex++] = 0;
    }
  }

  // Bytes 1 and 2 carry the frame ID. Apply valid frames in arrival order.
  if (writeIndex != rawPacketSize || raw[0] != protocolVersion || raw[3] != num_LED) return;
  if (crc8(raw, rawPacketSize - 1) != raw[rawPacketSize - 1]) return;

  uint8_t states[num_LED];
  for (int i = 0; i < num_LED; i += 2) {
    uint8_t packed = raw[headerSize + i / 2];
    states[i] = packed >> 4;
    states[i + 1] = packed & 0x0F;
  }
  for (int i = 0; i < num_LED; i++) {
    if ((states[i] >> 2) > 2 || (states[i] & 3) > 2) return;
  }
  for (int i = 0; i < num_LED; i++) {
    brightnessGrid[i] = states[i] >> 2;
    depthGrid[i] = depthIntensity[states[i] & 3];
  }
}

void receiveDepthData() {
  while (Serial.available()) {
    uint8_t byte = Serial.read();
    if (byte == 0) {
      if (!discardInputPacket) applyPacket();
      inputIndex = 0;
      discardInputPacket = false;
    } else if (!discardInputPacket) {
      if (inputIndex < encodedPacketSize) {
        inputBuffer[inputIndex++] = byte;
      } else {
        discardInputPacket = true;  // Ignore the rest of this frame through 0x00.
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
      // Shared clock: each LED uses its own level's half-period, without delays.
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
