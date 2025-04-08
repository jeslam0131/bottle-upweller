import RPi.GPIO as GPIO
import time

# Set the GPIO mode
GPIO.setmode(GPIO.BCM)

# Pin where the flow sensor is connected
FLOW_SENSOR_PIN = 13

# Set up the sensor pin as input
GPIO.setup(FLOW_SENSOR_PIN, GPIO.IN)

pulse_count = 0

# Function to count pulses
def count_pulse(channel):
    global pulse_count
    pulse_count += 1

# Attach the interrupt to the flow sensor pin
GPIO.add_event_detect(FLOW_SENSOR_PIN, GPIO.FALLING, callback=count_pulse)

try:
    while True:
        pulse_count = 0
        start_time = time.time()

        # Wait for 1 minute
        time.sleep(60)

        end_time = time.time()

        # Calculate the number of pulses in the last minute
        pulses_per_minute = pulse_count

        # Calculate the flow rate in L/min
        flow_rate = pulses_per_minute / 450.0  # 450 pulses per liter

        print(f"Flow rate: {flow_rate:.2f} L/min")

except KeyboardInterrupt:
    print("Program terminated.")
finally:
    GPIO.cleanup()
