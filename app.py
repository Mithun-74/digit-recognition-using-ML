import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# Load model once when app starts
model = tf.keras.models.load_model("mnist_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]
    img_bytes = base64.b64decode(data.split(",")[1])
    img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    prediction = np.argmax(model.predict(img))
    return jsonify({"digit": int(prediction)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))   # Render provides $PORT
    app.run(host="0.0.0.0", port=port)
