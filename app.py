import os
import requests  
import numpy as np
import tensorflow as tf
import time
import random
from flask import Flask, request, jsonify



app = Flask(__name__)


model_url = "https://www.dropbox.com/scl/fi/ehfy920bd4367pprwe9p6/trained_model.h5?rlkey=wkz3qlc82bym1m0e3jxxfivbs&st=jhgqhe7i&dl=0"

# Create the directory if it doesn't exist
# model_dir = os.path.dirname("trained_model.h5")
# model_dir ="trained_model.h5"


# try:
#   # Download the model
#   response = requests.get(model_url, stream=True)
#   response.raise_for_status()  # Raise an exception for non-200 status codes

#   # Write the model file
#   with open("trained_model.h5", 'wb') as f:
#     for chunk in response.iter_content(1024):
#       f.write(chunk)

#   # Load the model only if download is successful

#   print("Model loaded successfully!")

# except requests.exceptions.RequestException as e:
#   print(f"Error downloading model: {e}")
#   model = None  # Set model to None to indicate failure

# try:


#   os.system("wget -O trained_model.h5  https://www.dropbox.com/scl/fi/ehfy920bd4367pprwe9p6/trained_model.h5?rlkey=wkz3qlc82bym1m0e3jxxfivbs&st=jhgqhe7i&dl=0")
  
# except requests.exceptions.RequestException as e:
#   print(f"Error downloading model: {e}")
#   model = None  # Set model to None to indicate failure
model = tf.keras.models.load_model('trained_model.h5')


 
class_names=['NEGATIVE', 'POSITIVE']


def predict_image(image_path):
  print(image_path)
  img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256), color_mode="grayscale")
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

  predictions=[0,23,40]


  return jsonify({'predicted_class':predicted_class,"confidence":confidence})

# Start the Flask server in a new thread