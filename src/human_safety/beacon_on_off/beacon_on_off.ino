
#include <ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>

ros::NodeHandle  nh;

const int red_pin=8;
const int yellow_pin=12;
const int green_pin=13;
const int period_blink=1000;
const int time_without_msg=3000;
unsigned long msg_time = 0;  
unsigned long light_time = 0;  
unsigned long last_light_time = 0;
String  current_alert="none";
bool blinking_flag=true;
bool collision=false;

void messageCb( const std_msgs::String& data){
  current_alert=data.data;
  msg_time=millis();
}

ros::Subscriber<std_msgs::String> sub("visual_alerts", &messageCb );
std_msgs::Bool bool_msg;
ros::Publisher chatter("collision_detection", &bool_msg);

void setup()
{ 
  pinMode(red_pin, OUTPUT);
  pinMode(yellow_pin, OUTPUT);
  pinMode(green_pin, OUTPUT);
  attachInterrupt(digitalPinToInterrupt(2),PadReleased,CHANGE);
  //attachInterrupt(digitalPinToInterrupt(2),PadPressed,LOW); 
  nh.initNode();
  nh.advertise(chatter);
  nh.subscribe(sub);
  
}

void check_time()
{
  light_time = millis();
  if (light_time - last_light_time > period_blink)
  {
    last_light_time = light_time;
    if (blinking_flag==true){
     blinking_flag=false; 
    }
    else{
      blinking_flag=true;
    }
  }
  if (light_time- msg_time > time_without_msg){
    current_alert="none";
  }
}


void activation()
{
  check_time();
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
      }
      else{
        digitalWrite(green_pin, LOW);   // deactivate the green led
        digitalWrite(red_pin, LOW);   // deactivate the red led
        digitalWrite(yellow_pin, LOW);   // deactivate the yellow led
        
      }
  }
  if (current_alert=="red_blink"){
      if (blinking_flag==true){
        digitalWrite(green_pin, LOW);   // deactivate the green led
        digitalWrite(red_pin, HIGH);   // activate the red led
        digitalWrite(yellow_pin, LOW);   // deactivate the yellow led
      }
      else{
        digitalWrite(green_pin, LOW);   // deactivate the green led
        digitalWrite(red_pin, LOW);   // deactivate the red led
        digitalWrite(yellow_pin, LOW);   // deactivate the yellow led
        
      }
  }
  if (current_alert=="green_blink"){
      if (blinking_flag==true){
        digitalWrite(green_pin, HIGH);   // activate the green led
        digitalWrite(red_pin, LOW);   // deactivate the red led
        digitalWrite(yellow_pin, LOW);   // deactivate the yellow led
      }
      else{
        digitalWrite(green_pin, LOW);   // deactivate the green led
        digitalWrite(red_pin, LOW);   // deactivate the red led
        digitalWrite(yellow_pin, LOW);   // deactivate the yellow led
        
      }
  }
}

void PadReleased()          
{  
  if (collision==true){
   collision=false;                  
  }
  else{
    collision=true;
  }
   
}

//void PadPressed()          
//{                   
//   collision=true;
//}
  
void loop()
{ 
  activation();
  bool_msg.data = collision;
  chatter.publish( &bool_msg );
  nh.spinOnce();
  delay(1);
}
