import argparse
import base64
import json
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import os
import numpy as np
from config import *
from load_data import preprocess
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf

#tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED

@sio.on('telemetry')
def telemetry(sid, data):
	if data:
		# The current steering angle of the car
		steering_angle = float(data["steering_angle"])
		# The current throttle of the car
		throttle = float(data["throttle"])
		# The current speed of the car
		speed = float(data["speed"])
		# The current image from the center camera of the car
		imgString = data["image"]
		image = Image.open(BytesIO(base64.b64decode(imgString)))

		# frames incoming from the simulator are in RGB format
		image_array = cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2BGR)

		# perform preprocessing (crop, resize etc.)
		image_array = preprocess(frame_bgr=image_array)

		# add singleton batch dimension
		image_array = np.expand_dims(image_array, axis=0)

		# This model currently assumes that the features of the model are just the images. Feel free to change this.
		steering_angle = float(model.predict(image_array, batch_size=1))
		# lower the throttle as the speed increases
		# if the speed is above the current speed limit, we are on a downhill.
		# make sure we slow down first and then go back to the original max speed.

		global speed_limit

		if speed > speed_limit:
			speed_limit = MIN_SPEED  # slow down
		else:
			speed_limit = MAX_SPEED
		throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2


		print('{} {} {}'.format(steering_angle, throttle, speed))
		# save frame
		if args.image_folder != '':
			timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
			image_filename = os.path.join(args.image_folder, timestamp)
			image.save('{}.jpg'.format(image_filename))  
		#print(steering_angle, throttle)
		send_control(steering_angle, throttle)
	else:
		# NOTE: DON'T EDIT THIS.
		sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Remote Driving')
	'''
	parser.add_argument(
	    'model',
	    type=str,
	    help='Path to model h5 file. Model should be on the same path.'
	)
	'''
	parser.add_argument(
		'image_folder',
		type=str,
		nargs='?',
		default='',
		help='Path to image folder. This is where the images from the run will be saved.'
	)
	args = parser.parse_args()


	from keras.models import model_from_json

	# load model from json
	json_path ='pretrained/model.json'
	with open(json_path) as jfile:
		model = model_from_json(jfile.read())

	# load model weights
	# weights_path = os.path.join('checkpoints', os.listdir('checkpoints')[-1])
	weights_path = 'pretrained/model.hdf5'
	print('Loading weights: {}'.format(weights_path))
	model.load_weights(weights_path)

	# compile the model
	model.compile("adam", "mse")

	if args.image_folder != '':
		print("Creating image folder at {}".format(args.image_folder))
		if not os.path.exists(args.image_folder):
			os.makedirs(args.image_folder)
		else:
			shutil.rmtree(args.image_folder)
			os.makedirs(args.image_folder)
		print("RECORDING THIS RUN ...")
	else:
		print("NOT RECORDING THIS RUN ...")
	# wrap Flask application with engineio's middleware
	app = socketio.Middleware(sio, app)

	# deploy as an eventlet WSGI server
	eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
