//www.kuongshun.com

#include <Servo.h>  //servo library
Servo myservo;      // create servo object to control servo

int Echo = A4;  
int Trig = A5; 

#define ENA 5
#define ENB 6
#define IN1 7
#define IN2 8
#define IN3 9
#define IN4 11
#define carSpeed 200
int rightDistance = 0, leftDistance = 0, middleDistance = 0;

void forward(){ 
  analogWrite(ENA, carSpeed);
  analogWrite(ENB, carSpeed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  Serial.println("Forward");
}

void back() {
  analogWrite(ENA, carSpeed);
  analogWrite(ENB, carSpeed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  Serial.println("Back");
}

void left() {
  analogWrite(ENA, carSpeed);
  analogWrite(ENB, carSpeed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH); 
  Serial.println("Left");
}

void right() {
  analogWrite(ENA, carSpeed);
  analogWrite(ENB, carSpeed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  Serial.println("Right");
}

void stop() {
  digitalWrite(ENA, LOW);
  digitalWrite(ENB, LOW);
  Serial.println("Stop!");
} 

//Ultrasonic distance measurement Sub function
int Distance_test() {
  digitalWrite(Trig, LOW);   
  delayMicroseconds(2);
  digitalWrite(Trig, HIGH);  
  delayMicroseconds(20);
  digitalWrite(Trig, LOW);   
  float Fdistance = pulseIn(Echo, HIGH);  
  Fdistance= Fdistance / 58;       
  return (int)Fdistance;
}  

void setup() { 
  myservo.attach(3);  // attach servo on pin 3 to servo object
  Serial.begin(9600);     
  pinMode(Echo, INPUT);    
  pinMode(Trig, OUTPUT);  
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  stop();
} 

void loop() { 
    myservo.write(90);  //setservo position according to scaled value
    delay(500); 
    middleDistance = Distance_test();

    if(middleDistance <= 25) {     
      stop();
      delay(500);                         
      myservo.write(10);          
      delay(500);      
      rightDistance = Distance_test();
      
      delay(500);
      myservo.write(90);              
      delay(500);                                                  
      myservo.write(170);              
      delay(500); 
      leftDistance = Distance_test();
      
      delay(500);
      myservo.write(90);              
      delay(500);
      if(rightDistance > leftDistance) {
        stop();
        delay(100);
        back();
        delay(200);
        right();
        delay(300);
      }
      else if(rightDistance < leftDistance) {
        stop();
        delay(100);
        back();
        delay(200);
        left();
        delay(300);
      }
      else if((rightDistance <= 25) || (leftDistance <= 25)) {
        back();
        delay(200);
      }
      else {
        forward();
      }
    }  
    else {
        forward();
    }                     
}
