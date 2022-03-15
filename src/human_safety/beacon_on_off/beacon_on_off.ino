
#include <ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/String.h>
ros::NodeHandle  nh;

const int red_pin=8;
const int yellow_pin=12;
const int green_pin=13;
const int period_blink=1000;
String  current_alert="none";
bool blinking_flag=true;

void messageCb( const std_msgs::String& data){
  current_alert=data.data;
}

ros::Subscriber<std_msgs::String> sub("visual_alerts", &messageCb );

void setup()
{ 
  pinMode(red_pin, OUTPUT);
  pinMode(yellow_pin, OUTPUT);
  pinMode(green_pin, OUTPUT);
  nh.initNode();
  nh.subscribe(sub);
  
}

void activation()
{
  if (current_alert=="green"){
      digitalWrite(green_pin, HIGH);   // activate the green led
      digitalWrite(red_pin, LOW);   // deactivate the red led
      digitalWrite(yellow_pin, LOW);   // deactivate the yellow led
  }
  if (current_alert=="yellow"){
      digitalWrite(yellow_pin, HIGH);   // activate the yellow led
      digitalWrite(red_pin, LOW);   // deactivate the red led
      digitalWrite(green_pin, LOW);   // deactivate the green led
  }
  if (current_alert=="red"){
      digitalWrite(red_pin,HIGH);   // activate the red led
      digitalWrite(yellow_pin,LOW);   // deactivate the yellow led
      digitalWrite(green_pin,LOW);   // deactivate the green led
  }
  if (current_alert=="none"){
      digitalWrite(green_pin, LOW);   // deactivate the green led
      digitalWrite(red_pin, LOW);   // deactivate the red led
      digitalWrite(yellow_pin, LOW);   // deactivate the yellow led
  }
  if (current_alert=="yellow_blink"){
      if (blinking_flag==true){
        digitalWrite(green_pin, LOW);   // deactivate the green led
        digitalWrite(red_pin, LOW);   // deactivate the red led
        digitalWrite(yellow_pin, HIGH);   // activate the yellow led
        blinking_flag=false;
        delay(period_blink);
      }
      else{
        digitalWrite(green_pin, LOW);   // deactivate the green led
        digitalWrite(red_pin, LOW);   // deactivate the red led
        digitalWrite(yellow_pin, LOW);   // deactivate the yellow led
        blinking_flag=true;
        delay(period_blink);
        
      }
  }
  if (current_alert=="red_blink"){
      if (blinking_flag==true){
        digitalWrite(green_pin, LOW);   // deactivate the green led
        digitalWrite(red_pin, HIGH);   // activate the red led
        digitalWrite(yellow_pin, LOW);   // deactivate the yellow led
        blinking_flag=false;
        delay(period_blink);
      }
      else{
        digitalWrite(green_pin, LOW);   // deactivate the green led
        digitalWrite(red_pin, LOW);   // deactivate the red led
        digitalWrite(yellow_pin, LOW);   // deactivate the yellow led
        blinking_flag=true;
        delay(period_blink);
        
      }
  }
  if (current_alert=="green_blink"){
      if (blinking_flag==true){
        digitalWrite(green_pin, HIGH);   // activate the green led
        digitalWrite(red_pin, LOW);   // deactivate the red led
        digitalWrite(yellow_pin, LOW);   // deactivate the yellow led
        blinking_flag=false;
        delay(period_blink);
      }
      else{
        digitalWrite(green_pin, LOW);   // deactivate the green led
        digitalWrite(red_pin, LOW);   // deactivate the red led
        digitalWrite(yellow_pin, LOW);   // deactivate the yellow led
        blinking_flag=true;
        delay(period_blink);
        
      }
  }
}
  
void loop()
{ 
  activation();
  nh.spinOnce();
  delay(1);
}
