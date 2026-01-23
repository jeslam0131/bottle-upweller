import serial
import time

# CHANGE THIS to match your Arduino's port
# Windows: COM3, COM4, COM5...
# Mac/Linux: /dev/tty.usbmodemXXXX or /dev/ttyACM0
ser = serial.Serial('COM7', 115200, timeout=1)

time.sleep(2)  # wait for Arduino to reset


def send_command(cmd):
    ser.write((cmd + "\n").encode())
    print("Sent:", cmd)


print("Ready! Type FWD, REV, or STOP")
while True:
    user_input = input("Enter command: ").strip().upper()

    if user_input in ["FWD", "REV", "STOP"]:
        send_command(user_input)
    else:
        print("Invalid input. Try again.")
