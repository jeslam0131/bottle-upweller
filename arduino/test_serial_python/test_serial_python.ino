int dirPin = 12;   // direction pin
int pwmPin = 3;    // PWM pin
int brakePin = 9;  // brake pin

void setup() {
  pinMode(dirPin, OUTPUT);
  pinMode(pwmPin, OUTPUT);
  pinMode(brakePin, OUTPUT);

  digitalWrite(brakePin, LOW);  // release brake
  analogWrite(pwmPin, 0);       // start with motor off

  Serial.begin(115200);
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "FWD") {
      digitalWrite(dirPin, HIGH);
      digitalWrite(brakePin, LOW);
      analogWrite(pwmPin, 150); // full speed
    }

    else if (cmd == "REV") {
      digitalWrite(dirPin, LOW);
      digitalWrite(brakePin, LOW);
      analogWrite(pwmPin, 150); // full speed
    }

    else if (cmd == "STOP") {
      analogWrite(pwmPin, 0);
      digitalWrite(brakePin, HIGH);  // engage brake
    }
  }
}
