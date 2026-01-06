import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
DIR_PIN = 27
GPIO.setup(DIR_PIN, GPIO.OUT)

while True:
    GPIO.output(DIR_PIN, GPIO.HIGH)  # Reverse direction
    time.sleep(2)
    GPIO.output(DIR_PIN, GPIO.LOW)   # Forward direction
    time.sleep(2)
