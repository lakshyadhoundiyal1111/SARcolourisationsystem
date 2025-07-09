from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import cv2
import os
from werkzeug.utils import secure_filename
from PIL import Image
from model import build_colorization_model  # Import model builder

app = Flask(__name__)

# Set paths
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
WEIGHTS_PATH = 'colorization_model_weights.weights.h5'

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Build and load model
model = build_colorization_model((64, 64, 1))
model.load_weights(WEIGHTS_PATH)

# Image Preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # (64, 64, 1)
    img = np.expand_dims(img, axis=0)   # (1, 64, 64, 1)
    return img

# Postprocessing
def postprocess_output(pred):
    pred = pred[0]  # remove batch dim
    pred = (pred * 255).astype(np.uint8)
    return pred

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, filename)

            file.save(input_path)

            img_input = preprocess_image(input_path)
            pred_output = model.predict(img_input, verbose=0)
            color_img = postprocess_output(pred_output)

            Image.fromarray(color_img).save(output_path)

            return render_template("index.html", input_image=input_path, output_image=output_path)

    return render_template("index.html")
