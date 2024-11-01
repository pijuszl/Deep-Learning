import os
from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
import PIL
from io import BytesIO
import torch
from torchvision import transforms
from model import GJS_CNN

app = Flask(__name__)
CORS(app)

model = GJS_CNN()
model.load_state_dict(torch.load("model.pth"))
print(model)

model.eval()
classes = ["Goldfish", "Jellyfish", "Starfish"]


def preprocess_image(image):
    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3200, 0.3404, 0.3805], std=[0.2891, 0.2468, 0.2806]
            ),
        ]
    )

    img = PIL.Image.open(image)
    img = preprocess(img).unsqueeze(0)
    return img


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No image selected for uploading."}), 400

    try:
        img = preprocess_image(BytesIO(file.read()))

        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.softmax(outputs, dim=1)
            max_prob, preds = torch.max(probabilities, 1)
            pred_label = classes[preds.item()]
            pred_prob = max_prob.item()

        return jsonify({"prediction": pred_label, "probability": pred_prob})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
