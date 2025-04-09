//www.kuongshun.com

#include "IRremote.h"

int receiver = 12;
/*define channel enable output pins*/
#define ENA 5	  // Left  wheel speed
#define ENB 6	  // Right wheel speed
/*define logic control output pins*/
#define IN1 7	  // Left  wheel forward
#define IN2 8	  // Left  wheel reverse
#define IN3 9	  // Right wheel reverse
#define IN4 11	// Right wheel forward
#define carSpeed 200	// initial speed of car >=0 to <=255

IRrecv irrecv(receiver);
uint32_t last_decodedRawData = 0;

void translateIR() // takes action based on IR code received
{
  if (irrecv.decodedIRData.flags)// Check if it is a repeat IR code 
  {
    irrecv.decodedIRData.decodedRawData = last_decodedRawData;    //set the current decodedRawData to the last decodedRawData 
    Serial.println("REPEAT!");
  } else//output the IR code on the serial monitor
  {
    Serial.print("IR code:0x");
    Serial.println(irrecv.decodedIRData.decodedRawData, HEX);
  }
  switch (irrecv.decodedIRData.decodedRawData)//map the IR code to the remote key
  {
    case 0xB946FF00: Serial.println("UP");forward();break;
    case 0xEA15FF00: Serial.println("DOWN");back();break;
    case 0xBB44FF00: Serial.println("LEFT");left();break;
    case 0xBC43FF00: Serial.println("RIGHT");right();break;
    case 0xBF40FF00: Serial.println("OK");stop0();break;
    case 0xAD52FF00: Serial.println("0");    break;
    case 0xE916FF00: Serial.println("1");digitalWrite(13, HIGH);break;
    case 0xE619FF00: Serial.println("2");digitalWrite(13, LOW);break;
    case 0xF20DFF00: Serial.println("3");    break;
    case 0xF30CFF00: Serial.println("4");    break;
    case 0xE718FF00: Serial.println("5");    break;
    case 0xA15EFF00: Serial.println("6");    break;
    case 0xF708FF00: Serial.println("7");    break;
    case 0xE31CFF00: Serial.println("8");    break;
    case 0xA55AFF00: Serial.println("9");    break;
    case 0xBD42FF00: Serial.println("*");    break;
    case 0xB54AFF00: Serial.println("#");    break;
    default:
    Serial.println(" other button   ");
  }// End Case  
  last_decodedRawData = irrecv.decodedIRData.decodedRawData;//store the last decodedRawData
  delay(500); // Do not get immediate repeat
} //END translateIR

 void forward(){ 
  digitalWrite(ENA,HIGH);
  digitalWrite(ENB,HIGH);
  digitalWrite(IN1,HIGH);
  digitalWrite(IN2,LOW);
  digitalWrite(IN3,LOW);
  digitalWrite(IN4,HIGH);
  Serial.println("go forward!");
}
void back(){
  digitalWrite(ENA,HIGH);
  digitalWrite(ENB,HIGH);
  digitalWrite(IN1,LOW);
  digitalWrite(IN2,HIGH);
  digitalWrite(IN3,HIGH);
  digitalWrite(IN4,LOW);
  Serial.println("go back!");
}
void left(){
  analogWrite(ENA,carSpeed);
  analogWrite(ENB,carSpeed);
  digitalWrite(IN1,LOW);
  digitalWrite(IN2,HIGH);
  digitalWrite(IN3,LOW);
  digitalWrite(IN4,HIGH); 
  Serial.println("go left!");
}
void right(){
  analogWrite(ENA,carSpeed);
  analogWrite(ENB,carSpeed);
  digitalWrite(IN1,HIGH);
  digitalWrite(IN2,LOW);
  digitalWrite(IN3,HIGH);
  digitalWrite(IN4,LOW);
  Serial.println("go right!");
}
void stop0(){
  digitalWrite(ENA, LOW);
  digitalWrite(ENB, LOW);
  Serial.println("STOP!");  
}

void setup() {
  Serial.begin(9600);
  pinMode(IN1,OUTPUT);
  pinMode(IN2,OUTPUT);
  pinMode(IN3,OUTPUT);
  pinMode(IN4,OUTPUT);
  pinMode(ENA,OUTPUT);
  pinMode(ENB,OUTPUT);
  pinMode(13, OUTPUT);
  stop0();
  irrecv.enableIRIn();// Start the receiver  
}

void loop() {
  if (irrecv.decode()) // have we received an IR signal?
  {
    translateIR();
    irrecv.resume(); // receive the next value
  }
} 
