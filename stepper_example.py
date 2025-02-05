import RPi.GPIO as GPIO
import time

# Pin definitions
ENA_PIN = 22  # ENA-
DIR_PIN = 17  # DIR-
PUL_PIN = 17  # PUL-

# Setup GPIO
GPIO.setmode(GPIO.BOARD)  # Using physical pin numbering
GPIO.setup(ENA_PIN, GPIO.OUT)
GPIO.setup(DIR_PIN, GPIO.OUT)
GPIO.setup(PUL_PIN, GPIO.OUT)

# Enable the driver by setting ENA- low
GPIO.output(ENA_PIN, GPIO.LOW)

# Function to perform a step
def step_motor():
    GPIO.output(PUL_PIN, GPIO.HIGH)  # Pulse HIGH
    time.sleep(0.001)  # Small delay
    GPIO.output(PUL_PIN, GPIO.LOW)   # Pulse LOW
    time.sleep(0.001)  # Small delay

# Function to rotate motor in one direction
def rotate_motor(direction, steps):
    # Set direction pin (HIGH for one direction, LOW for the other)
    if direction == "clockwise":
        GPIO.output(DIR_PIN, GPIO.HIGH)
    else:
        GPIO.output(DIR_PIN, GPIO.LOW)

    # Step the motor
    for _ in range(steps):
        step_motor()

# Main program
try:
    # Rotate motor clockwise for 200 steps
    print("Rotating clockwise...")
    rotate_motor("clockwise", 200)
    
    # Wait before reversing direction
    time.sleep(1)

    # Rotate motor counterclockwise for 200 steps
    print("Rotating counterclockwise...")
    rotate_motor("counterclockwise", 200)

    # Clean up
    GPIO.cleanup()

except KeyboardInterrupt:
    print("Program interrupted")
    GPIO.cleanup()
