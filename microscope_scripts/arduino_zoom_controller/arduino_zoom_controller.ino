#include <Servo.h>

Servo myServo;
const int SERVO_PIN = 11;

int moveServo(Servo &myServo, int target_angle, int degPerSecond=30) {
  int curr_angle = myServo.read();

  int milliseconds_per_deg = 1000 / degPerSecond;

  if (target_angle > curr_angle) {
    for (int i=curr_angle; i<=target_angle; i++) {
      myServo.write(i);
      delay(milliseconds_per_deg);
    }
  } else if (target_angle < curr_angle) {
    for (int i=curr_angle; i>=target_angle; i--) {
      myServo.write(i);
      delay(milliseconds_per_deg);
    }
  }
  
  return target_angle;
}

bool isValidInt(String s) {
  for (int i = 0; i < s.length(); i++) {
    if (i == 0 && s[i] == '-') continue;  // allow negative sign
    if (!isDigit(s[i])) return false;
  }
  return true;
}

void setup() {
  Serial.begin(9600);
  myServo.attach(SERVO_PIN);
  myServo.write(90); // 90 is starting angle always
}

void loop() {
  // if (Serial.available()) {
  //   int angle = Serial.parseInt();
  //   int result = -1;
  //   if (angle >= 0 && angle <= 180) {
  //     result = moveServo(myServo, angle, 60);
  //   }

  //   // once complete, send success message back
  //   Serial.println(result);
  // }

  if (Serial.available()) {
    String incoming = Serial.readStringUntil('\n');
    incoming.trim();  // removes whitespace/garbage characters

    // check it's actually a number before parsing
    if (incoming.length() > 0 && isValidInt(incoming)) {
      int angle = incoming.toInt();
      int result = -1;

      if (angle >= 0 && angle <= 180) {
        result = moveServo(myServo, angle, 60);
      }

      Serial.println(result);
    }
  }
}