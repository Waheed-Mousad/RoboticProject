
#include <Servo.h>

// === Pin Definitions ===
const int IR_L_PIN = 2;
const int IR_M_PIN = 4;
const int IR_R_PIN = 10;
const int TRIG_PIN = A5;
const int ECHO_PIN = A4;
const int SERVO_PIN = 12;

const int ENA = 5;
const int ENB = 6;
const int IN1 = 7;
const int IN2 = 8;
const int IN3 = 9;
const int IN4 = 11;

// === Globals ===
int leftOffset = 44;    // default calibration
int rightOffset = -44;
char currentCommand = 's'; // 's' = stop
Servo servoMotor;
#define carSpeed 200
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
  digitalWrite(ENA, LOW);
  digitalWrite(ENB, LOW);
}

void forward() {
  analogWrite(ENA, carSpeed+leftOffset);
  analogWrite(ENB, carSpeed+rightOffset);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  
}

void back() {
  analogWrite(ENA, carSpeed+leftOffset);
  analogWrite(ENB, carSpeed+rightOffset);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  
}

void left() {
  analogWrite(ENA, carSpeed+leftOffset);
  analogWrite(ENB, carSpeed+rightOffset);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  
}

void right() {
  analogWrite(ENA, carSpeed+leftOffset);
  analogWrite(ENB, carSpeed+rightOffset);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  
}

long measureDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000); // 30ms timeout
  long dist = duration / 58;
  Serial.println(dist);
  return (dist == 0 || dist > 400) ? 400 : dist;
}

void handleCommand(char cmd) {
  // Stop motors first - ensures clean transition between commands
  stopMotors();
  delayMicroseconds(50);
  
  currentCommand = cmd;
  switch (cmd) {
    case 'f': forward(); break;
    case 'b': back(); break;
    case 'l': left(); break;
    case 'r': right(); break;
    case 's': stopMotors(); break;
  }
}


// === Loop ===
void loop() {
  // Process incoming commands with improved buffering
  if (Serial.available() > 0) {
    // Wait a bit to receive the full command
    delay(5); 
    
    char cmd = Serial.read();
    if (cmd == 'f' || cmd == 'b' || cmd == 'l' || cmd == 'r' || cmd == 's') {
      handleCommand(cmd);
    } else if (cmd == 'L' || cmd == 'R') {
      // Make sure we have a full int value
      delay(10);
      int val = Serial.parseInt();
      if (cmd == 'L') leftOffset = val;
      if (cmd == 'R') rightOffset = val;
      
      // Immediately apply new calibration
      handleCommand(currentCommand);
    }
    
    // Clear any extra characters in buffer
    while (Serial.available() > 0) {
      Serial.read();
    }
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