
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

const int num_rows = 4;
const int num_cols = 4;
const int rowPins[num_rows] = {23, 19, 18, 5};
const int colChannels[num_cols] = {8, 9, 10, 11};

int num_LED = num_rows*num_cols;

int depthGrid[16] = {0};

char inputBuffer[128];
int inputIndex = 0;

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

      int newValues[num_LED];
      int count = 0;

      char* token = strtok(inputBuffer, ",");
      bool valid = true;

      while (token != nullptr && count < num_LED) {
        char* endPtr;
        long v = strtol(token, &endPtr, 10);

        if (endPtr == token || *endPtr != '\0' ||
            v < 0 || v > 100) {
          valid = false;
          break;
        }

        newValues[count++] = (int)v;
        token = strtok(nullptr, ",");
      }

      if (valid && count == num_LED && token == nullptr) {
        for (int i = 0; i < num_LED; i++) {
          depthGrid[i] = newValues[i];
        }
      }

      inputIndex = 0;
    }
    else if (ch != '\r') {
      if (inputIndex < (int)sizeof(inputBuffer) - 1) {
        inputBuffer[inputIndex++] = ch;
      } else {
        inputIndex = 0;  // Discard an oversized packet.
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
      setColumnBrightness(
        colChannels[col],
        depthGrid[ledIndex]
      );
      if(depthGrid[ledIndex]>0){flag_turn_on_row = 1;}
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
