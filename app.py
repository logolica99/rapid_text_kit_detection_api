import os
import requests  
import numpy as np
import tensorflow as tf
import time
import random
from flask import Flask, request, jsonify



app = Flask(__name__)


model = tf.keras.models.load_model('trained_model_robust.h5')


 
class_names=['NEGATIVE', 'POSITIVE']


def predict_image(image_path):
  print(image_path)
  img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128), color_mode="grayscale")
  img_array = tf.keras.preprocessing.image.img_to_array(img)

  img_tensor = tf.convert_to_tensor(img_array)
  img_array = tf.keras.preprocessing.image.img_to_array(img_tensor.numpy())
  img_array = tf.expand_dims(img_array, 0)

  predictions = model.predict(img_array)

  predicted_class = class_names[np.argmax(predictions[0])]
  confidence = round(100 * (np.max(predictions[0])), 2)
  return predicted_class,confidence

# Define Flask routes
@app.route("/")
def index():
    return "Hello!"


@app.route('/predict_coordinates', methods=['POST'])
def predict():
  # Get the image link from the request body
  data = request.get_json()
  if 'image_link' not in data:
    return jsonify({'error': 'Missing image_link in request body'}), 400

  image_link = data['image_link']


  # Download the image
  try:
    response = requests.get(image_link, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
  except requests.exceptions.RequestException as e:
    return jsonify({'error': f'Failed to download image: {e}'}), 400

  current_time_seconds = int(time.time())
  random_number1 = random.randint(0, 99)
  random_number2 = random.randint(0, 99)
  img_file_name = f"{current_time_seconds}{random_number1:02d}{random_number2:02d}.jpg"



  # Save the image to a temporary file
  with open(img_file_name, 'wb') as f:
    for chunk in response.iter_content(1024):
      f.write(chunk)

  predicted_class,confidence = predict_image(img_file_name)

  os.remove(img_file_name)


  return jsonify({'predicted_class':predicted_class,"confidence":confidence})

