import RPi.GPIO as GPIO
import time

# Set up the GPIO mode
GPIO.setmode(GPIO.BCM)

# Define the GPIO pins for the TB6600 connections
ENA_PIN = 22
DIR_PIN = 27
PUL_PIN = 17

# Set up the pins as outputs
GPIO.setup(ENA_PIN, GPIO.OUT)
GPIO.setup(DIR_PIN, GPIO.OUT)
GPIO.setup(PUL_PIN, GPIO.OUT)

# Enable the driver
GPIO.output(ENA_PIN, GPIO.HIGH)  # HIGH to enable the driver

# Set direction (HIGH = clockwise, LOW = counterclockwise)
GPIO.output(DIR_PIN, GPIO.LOW)  # Change to GPIO.HIGH for reverse direction

# Function to make the motor spin
def spin_motor(steps, delay=0.0000001):
    print(f"Spinning motor for {steps} steps...")
    for _ in range(steps):
        GPIO.output(PUL_PIN, GPIO.HIGH)
        time.sleep(delay)  # Pulse width duration
        GPIO.output(PUL_PIN, GPIO.LOW)
        time.sleep(delay)  # Pulse width duration

# Forward direction (clockwise)
def forward():
    print("Moving Forward")
    GPIO.output(DIR_PIN, GPIO.LOW)
    spin_motor(1000)  # 5000 steps, adjust based on your setup
    print("Forward movement completed.")

# Reverse direction (counterclockwise)
def reverse():
    print("Moving Backward")
    GPIO.output(DIR_PIN, GPIO.HIGH)
    spin_motor(1000)  # 5000 steps, adjust based on your setup
    print("Backward movement completed.")

try:
    # Spin motor forward and backward for a set number of cycles
    for cycle in range(10):  # Number of cycles
        forward()
        reverse()

finally:
    # Cleanup GPIO settings after use
    GPIO.cleanup()
    print("Motor control complete.")
