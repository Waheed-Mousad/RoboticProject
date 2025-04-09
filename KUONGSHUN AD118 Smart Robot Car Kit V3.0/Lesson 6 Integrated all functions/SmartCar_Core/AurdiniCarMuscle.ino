#include <Servo.h>

// === Pin Definitions ===
const int IR_L_PIN = 2;
const int IR_M_PIN = 3;
const int IR_R_PIN = 4;
const int TRIG_PIN = A1;
const int ECHO_PIN = A0;
const int SERVO_PIN = 12;

const int ENA = 5;
const int ENB = 6;
const int IN1 = 7;
const int IN2 = 8;
const int IN3 = 9;
const int IN4 = 10;

// === Globals ===
int leftOffset = 44;    // default calibration
int rightOffset = -44;
char currentCommand = 's'; // 's' = stop
Servo servoMotor;

// === Setup ===
void setup() {
  Serial.begin(115200);

  pinMode(IR_L_PIN, INPUT);
  pinMode(IR_M_PIN, INPUT);
  pinMode(IR_R_PIN, INPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  servoMotor.attach(SERVO_PIN);
  servoMotor.write(90); // center

  stopMotors();
}

// === Helper Functions ===
void stopMotors() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void setMotor(bool leftFwd, bool rightFwd, int baseSpeed = 150) {
  int ls = constrain(baseSpeed + leftOffset, 0, 255);
  int rs = constrain(baseSpeed + rightOffset, 0, 255);

  digitalWrite(IN1, leftFwd ? HIGH : LOW);
  digitalWrite(IN2, leftFwd ? LOW : HIGH);
  digitalWrite(IN3, rightFwd ? HIGH : LOW);
  digitalWrite(IN4, rightFwd ? LOW : HIGH);

  analogWrite(ENA, ls);
  analogWrite(ENB, rs);
}

long measureDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000); // 30ms timeout
  long dist = duration / 58;
  return (dist == 0 || dist > 400) ? 400 : dist;
}

void handleCommand(char cmd) {
  currentCommand = cmd;
  switch (cmd) {
    case 'f': setMotor(true, true); break;
    case 'b': setMotor(false, false); break;
    case 'l': setMotor(false, true); break;
    case 'r': setMotor(true, false); break;
    case 's': stopMotors(); break;
  }
}

// === Loop ===
void loop() {
  // Process incoming commands
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'f' || cmd == 'b' || cmd == 'l' || cmd == 'r' || cmd == 's') {
      handleCommand(cmd);
    } else if (cmd == 'L' || cmd == 'R') {
      int val = Serial.parseInt();
      if (cmd == 'L') leftOffset = val;
      if (cmd == 'R') rightOffset = val;
    }
    while (Serial.peek() == '\n' || Serial.peek() == '\r') Serial.read(); // flush
  }

  // Read sensors
  long distance = measureDistance();
  int irL = digitalRead(IR_L_PIN);
  int irM = digitalRead(IR_M_PIN);
  int irR = digitalRead(IR_R_PIN);

  // Send data: DISTANCE, IR_L, IR_M, IR_R
  Serial.print(distance);
  Serial.print(',');
  Serial.print(irL ? 1 : 0);
  Serial.print(',');
  Serial.print(irM ? 1 : 0);
  Serial.print(',');
  Serial.println(irR ? 1 : 0);

  delay(50);  // ~20Hz loop
}
