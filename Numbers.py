# TensorFlow Cereal Sorter
# Courtesy of the Stevens Robotics Club AI Workshop
# Created on February 11, 2024
# Created by: Mehmet Bertan Tarakcioglu

# Concept based on Google's Project, with updated code!
# Check out the original project:
# https://coral.ai/projects/teachable-sorter#project-intro

# IMPORTANT!
# Requires Raspberry Pi OS Bullseye 2023-02-21 on a
# Raspberry Pi 4 with a Camera Module V3 and Coral USB Accelerator
# as it is the newest version that still supports PyCoral and
# picamera2 at the same time without workarounds!

# If you cloned this from GitHub, check the bundled README.MD for
# details or refer to the original repository via this link:
# https://github.com/BertanT/SRC-Cereal-Sorter/blob/main/README.md


import argparse
import board
import cv2
import digitalio
import neopixel
import pwmio
import PIL
import numpy as np

from adafruit_motor import servo
from os import path
from picamera2 import Picamera2
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, classify
from sys import path as PATH
from time import sleep

# Default file paths (appending to current working directory)
DEFAULT_MODEL_PATH = 'model.tflite'
DEFAULT_LABEL_PATH = 'labels.txt'

# Pin configuration for electronics
# Not all pins support NeoPixel! Check this link for more info:
#   https://learn.adafruit.com/neopixels-on-raspberry-pi/raspberry-pi-wiring
VIBRATION_MOTOR_PIN = board.D6
NEOPIXEL_PIN = board.D21
SERVO_PIN = board.D12  # Must be a PWM pin!

# NeoPixel Config
NEOPIXEL_COUNT = 8
NEOPIXEL_ORDER = neopixel.GRB  # Change if you have RGBW!
NEOPIXEL_BRIGHTNESS = 1  # Between 0 and 1

# Servo config:
# May need to experiment with SERVO_MIN_PULSE and SERVO_MAX_PULSE to get accurate angles
#   on your specific model of servo motor. Check this link for more info:
#   https://learn.adafruit.com/using-servos-with-circuitpython/high-level-servo-control
SERVO_PWM_DUTY_CYCLE = 2 ** 15  # Recommended
SERVO_PWM_FREQ = 50  # Recommended
SERVO_MIN_PULSE = 650  # Works okay with Tower Pro SG92R
SERVO_MAX_PULSE = 2350  # Works okay with Tower Pro SG92R


def classify_image(interpreter, labels, image):
    # Get the input size of the model from the interpreter
    size = common.input_size(interpreter)

    # Variable to store the image after processing
    processed_image = image
    # Flip the image
    processed_image = cv2.flip(processed_image, 1)
    # Convert the image to RGB format
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    # Resize the image in accordance with the model input size
    processed_image = cv2.resize(processed_image, size, fx=0, fy=0,
                                 interpolation=cv2.INTER_CUBIC)

    # Feed the processed image into the model's input tensor
    common.set_input(interpreter, processed_image)

    # Invoke the interpreter to perform classification
    interpreter.invoke()

    # Return the ID (label in this case) of the best matching image class
    top_obj = classify.get_classes(interpreter, top_k=1)[0]

    return labels.get(top_obj.id)


def main():
    # Set up CLI arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        help='Coral TPU TensorFlow Lite model path. Default: ./{}'
                        .format(DEFAULT_MODEL_PATH),
                        default=path.join(PATH[0], DEFAULT_MODEL_PATH))
    parser.add_argument('--labels',
                        help='Label file path. Default: ./{}'
                        .format(DEFAULT_LABEL_PATH),
                        default=path.join(PATH[0], DEFAULT_LABEL_PATH))
    args = parser.parse_args()

    # Greet the user
    print('Welcome to the Stevens Robotics Club Cereal Sorter :)')
    print('\nLoading model: {} with labels: {}...'
          .format(args.model, args.labels))

    # Set up the digital output for the vibration motor
    vibration_motor = digitalio.DigitalInOut(VIBRATION_MOTOR_PIN)
    vibration_motor.direction = digitalio.Direction.OUTPUT
    # Make sure no cereal is dispensed before classification begins
    vibration_motor.value = False

    # Initialize, set up, and power up the NeoPixels to provide light for the camera
    pixels = neopixel.NeoPixel(NEOPIXEL_PIN,
                               NEOPIXEL_COUNT,
                               brightness=NEOPIXEL_BRIGHTNESS,
                               pixel_order=NEOPIXEL_ORDER)
    pixels.fill((255, 255, 255))  # White light at full brightness
    pixels.show()

    # Set up the PWM pin, initialize, and home the servo motor
    servo_pwm = pwmio.PWMOut(SERVO_PIN,
                             duty_cycle=SERVO_PWM_DUTY_CYCLE,
                             frequency=SERVO_PWM_FREQ)
    servo_motor = servo.Servo(servo_pwm,
                              min_pulse=SERVO_MIN_PULSE,
                              max_pulse=SERVO_MAX_PULSE)
    servo_motor.angle = 90

    # Create interpreter for image recognition with the input model
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    # Initialize model labels
    labels = read_label_file(args.labels)

    # Initialize and start the Raspberry Pi Camera Module
    camera = Picamera2()
    camera.start()

    print('Started object classification:')

    debounce_counter = 0
    prev_object_id = ''
    try:
        while True:

            # Capture image from camera as an array
            image = camera.capture_image()
            image = image.convert('L')
            image = image.resize((28, 28))
            image = np.array(image)
            image = image / 255.0

            object_id = classify_image(interpreter, labels, image)

            print("\n " + object_id)
            sleep(1)

    except KeyboardInterrupt:
        print("\nStopping cereal, homing servo, and exiting script. See you later!")
        vibration_motor.value = False
        servo_motor.angle = 90
        sleep(1)


if __name__ == '__main__':
    main()
