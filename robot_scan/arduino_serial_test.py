import time

import serial

# Change /dev/ttyUSB0 to your actual port
ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)
time.sleep(2)  # Wait for Arduino to reset after connection


def wait_for_angle():
    while True:
        line = ser.readline().decode("utf-8").strip()
        if line.lstrip("-").isdigit():  # Make sure it's actually a number
            return int(line)


def move_servo(angle):
    if 0 <= angle <= 180:
        ser.write(f"{angle}\n".encode())
        print(f"Sent angle: {angle}")


# Example usage
while True:
    angle = input("Enter angle (0-180) or 'exit' to quit: ")
    if angle.lower() == "exit":
        break
    move_servo(int(angle))
ser.close()
