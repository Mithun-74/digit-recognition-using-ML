# app.py
import os
import base64
import io
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("mnist_model.h5")

def preprocess_image(image_data):
    # Decode base64 image
    image_data = image_data.split(",")[1]  # remove "data:image/png;base64,"
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # convert to grayscale

    # Invert colors (canvas = black background, MNIST = white background)
    img = ImageOps.invert(img)

    # Resize to 28x28 (MNIST format)
    img = img.resize((28, 28))

    # Convert to numpy array
    img_array = np.array(img) / 255.0

    # Reshape for model (1, 28, 28)
    img_array = img_array.reshape(1, 28, 28)

    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = data["image"]

    processed = preprocess_image(image_data)
    predictions = model.predict(processed)
    digit = int(np.argmax(predictions))

    return jsonify({
        "prediction": digit,
        "probabilities": predictions.tolist()
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render injects $PORT
    app.run(host="0.0.0.0", port=port, debug=False)

