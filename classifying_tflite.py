# Importing necessary libraries
# For performing array functions
import numpy as np
# Pyserial library to read data form the serial ports
import serial
# Tensorflow.lite to use the tflite model
import tensorflow.lite as tflite
# Mouse library for mouse control
import mouse
# Keyboard library for keyboard control
import keyboard
# Subprocess library for opening the pdf reader (Adobe acrobat)
import subprocess

GESTURES = ['up', 'down', 'circle', 'counter', 'pan']

# Load the tflite model
interpreter = tflite.Interpreter(model_path='gesture_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# takes 119 samples from the accelerometer
num_samples = 119
# specify serial port
ser = serial.Serial('COM3', 9600)

# Open system pdf reader (specify application location)
subprocess.Popen('C:/Program Files (x86)/Adobe/Reader 10.0/Reader/AcroRd32.exe')

# initialize num to 0
num = 0

# Main function loop
while True:
    # initialize samples_read to 0
    samples_read = 0
    # initialize a 1D array to store normalized data
    all_data = []
    # loop to read 119 samples
    while samples_read < num_samples:
        # store serial data in data and split with comma
        data = ser.readline().decode().strip()
        values = data.split(',')

        # normalizing accelerometer data and incrementing samples_read
        acc = [float(value) for value in values]
        normalized_data = np.add(acc, 4.0) / 8.0
        imu_data = normalized_data
        samples_read += 1

        # appending normalized data to all_data
        all_data = np.append(all_data, imu_data)
        # convert float64 data in to float32 data
        all_data_float = all_data.astype(np.float32)
        # reshaping data to fit the tflite model
        all_data_float = all_data_float.reshape(1, -1)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], all_data_float)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # finding the gesture with the highest confidence
    for i in range(len(GESTURES)):
        gesture = output[0][i]
        if output[0][i] > gesture:
            gesture = output[0][i]
            num = i

    # print output
    max_index = np.argmax(output)
    print(GESTURES[max_index])

    # performing keyboard and mouse functions based on the obtained gestures
    gesture = str(GESTURES[max_index])
    if gesture == 'up':
        # scroll up
        mouse.wheel(1)
        mouse.wheel(1)
        mouse.wheel(1)
        mouse.wheel(1)
        mouse.wheel(1)

    if gesture == 'down':
        # scroll down
        mouse.wheel(-1)
        mouse.wheel(-1)
        mouse.wheel(-1)
        mouse.wheel(-1)
        mouse.wheel(-1)

    if gesture == 'circle':
        # zoom in
        keyboard.press('ctrl')
        keyboard.press_and_release('+')
        keyboard.release('ctrl')

    if gesture == 'counter':
        # zoom out
        keyboard.press('ctrl')
        keyboard.press_and_release('-')
        keyboard.release('ctrl')

    if gesture == 'pan':
        # rotate the screen clockwise
        keyboard.press('ctrl')
        keyboard.press('shift')
        keyboard.press_and_release('+')
        keyboard.release('shift')
        keyboard.release('ctrl')

    # delete all data in all_data to store accelerometer data for the next loop
    del all_data
