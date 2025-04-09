//www.kuongshun.com

#include "IRremote.h"
int receiver = 12; 
IRrecv irrecv(receiver);     
uint32_t last_decodedRawData = 0;

void translateIR() 
{

  if (irrecv.decodedIRData.flags)
  {
    irrecv.decodedIRData.decodedRawData = last_decodedRawData;
    Serial.println("REPEAT!");
  } else
  {
    Serial.print("IR code:0x");
    Serial.println(irrecv.decodedIRData.decodedRawData, HEX);
  }
  switch (irrecv.decodedIRData.decodedRawData)
  {
    case 0xB946FF00: Serial.println("UP"); break;
    case 0xEA15FF00: Serial.println("DOWN"); break;
    case 0xBB44FF00: Serial.println("LEFT"); break;
    case 0xBC43FF00: Serial.println("RIGHT");    break;
    case 0xBF40FF00: Serial.println("OK");    break;
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
  }
  last_decodedRawData = irrecv.decodedIRData.decodedRawData;
  delay(100); 
}

void setup() 
{
  Serial.begin(9600);
  pinMode(13, OUTPUT);
  Serial.println("IR Receiver Button Decode");
  irrecv.enableIRIn(); 
}

void loop()
{
  if (irrecv.decode())
  {
    translateIR();
    irrecv.resume(); 
  }
}
