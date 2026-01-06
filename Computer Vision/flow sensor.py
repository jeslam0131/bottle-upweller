import serial
import time

# Configure the serial port
port = "/dev/ttyUSB0"  # Replace with your actual serial port
baudrate = 9600  # Replace with your device's baud rate

try:
    ser = serial.Serial(port, baudrate, timeout=1)
    time.sleep(2)  # Give the device time to initialize/reset if needed
    ser.flushInput()

    print("Listening on serial port...")
    while True:
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8', errors='replace').rstrip()
            print(data)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Interrupted by user")
except serial.SerialException as e:
    print(f"Serial error: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed")
