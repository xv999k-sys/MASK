
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)


model = tf.keras.models.load_model(r"C:\Users\admin\Desktop\project\mask_cnn_model(karam_kv).h5")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        label = "without_mask"
    else:
        label = "with_mask"

    print("Result Sent To Frontend:", label)

    return jsonify({"result": label})

    
    

if __name__ == "__main__":
    app.run(debug=True)
