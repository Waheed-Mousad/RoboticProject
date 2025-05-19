// === Pin Definitions ===
const int IR_L_PIN = 2; 
const int IR_M_PIN = 4;
const int IR_R_PIN = 10;
const int TRIG_PIN = A5;
const int ECHO_PIN = A4;

// motors specific pins
const int ENA = 5;
const int ENB = 6;
const int IN1 = 7;
const int IN2 = 8;
const int IN3 = 9;
const int IN4 = 11;

// === Globals ===
int leftOffset = 0;    // default calibration
int rightOffset = 0;
char currentCommand = 's'; // 's' = stop
int carSpeed = 150; // defualt
// === Setup ===
void setup() {
  Serial.begin(9600);

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

  stopMotors();
}

// === Helper Functions ===
// the following helper functions are taken from 
// Kuongshun Electronic Limited. (n.d.). KUONGSHUN AD118 Smart Robot Car Kit V3.0 [ZIP archive]. Dropbox. https://www.dropbox.com/scl/fi/02afmsvjzr08bmra9n9n6/KUONGSHUN-AD118-Smart-Robot-Car-Kit-V3.0.zip?rlkey=0opnshepwkli75gusa3ru3y0a&e=1&st=mfzl8c2e&dl=0
// with limited changes
void stopMotors() {
  digitalWrite(ENA, LOW); //disable left motors
  digitalWrite(ENB, LOW); //disable right motors
}

/*
for the following functions, they have this simple
table to explain them
note that X = don't care

| ENB, ENA     | IN1 | IN2 | IN3 | IN4 | DC MOTOR STATUS  |
|--------------|-----|-----|-----|-----|------------------|
| 0            |  X  |  X  |  X  |  X  | STOP             |
| 1 < ; < 255  |  1  |  0  |  0  |  1  | Forward          |
| 1 < ; < 255  |  0  |  1  |  1  |  0  | Back             |
| 1 < ; < 255  |  0  |  1  |  0  |  1  | Left             |
| 1 < ; < 255  |  1  |  0  |  1  |  0  | Right            |

*/
void forward() {
  analogWrite(ENA, carSpeed+leftOffset);  // set speed for left motor
  analogWrite(ENB, carSpeed+rightOffset); // set speed for right motors
  digitalWrite(IN1, HIGH); // left motors forward
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); // right motors forward
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
// === Measures distance using ultrasonic sensor (HC-SR04) ===
long measureDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH); // Send 10us pulse
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000); // Wait for echo, max 30ms
  long dist = duration / 58; // Convert to cm

   // Cap distance between 0 and 400 cm
  return (dist == 0 || dist > 400) ? 400 : dist;
}

void handleCommand(char cmd) {
  // Stop motors first - ensures clean transition between commands
  stopMotors();
  delayMicroseconds(50);
  // based on the recieved command, call the following function
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
    } else if (cmd == 'L' || cmd == 'R' || cmd == 'S') {
      // Make sure we have a full int value
      delay(10);
      int val = Serial.parseInt();
      if (cmd == 'L') leftOffset = val;
      if (cmd == 'R') rightOffset = val;
      if (cmd == 'S') carSpeed = val;
      
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
  Serial.print(irL);
  Serial.print(',');
  Serial.print(irM);
  Serial.print(',');
  Serial.println(irR);

  delay(20);  // ~50Hz loop
}