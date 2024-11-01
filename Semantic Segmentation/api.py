from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64

app = Flask(__name__)
CORS(app)

# Load your TensorFlow model
model = load_model("model.h5")
model.summary()


def mask_to_image(mask):
    colors = np.array(
        [
            [0, 0, 0, 0],  # Class 0: Background (transparent)
            [255, 0, 0, 255],  # Class 1: Airplane (red)
            [0, 255, 0, 255],  # Class 2: Car (green)
            [0, 0, 255, 255],  # Class 3: Person (blue)
        ]
    )
    height, width = mask.shape
    color_mask = np.zeros((height, width, 4), dtype=np.uint8)
    for label in range(len(colors)):
        color_mask[mask == label] = colors[label]
    return color_mask


@app.route("/", methods=["GET"])
def home():
    """Render the home page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    try:
        # Read the image file and prepare for prediction
        image_bytes = file.read()
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(
            image, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        image = tf.image.rgb_to_grayscale(image)

        print("Original image shape:", image.shape)
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)
        elif len(image.shape) == 4 and image.shape[0] != 1:
            image = image[:1]
        else:
            print(
                "Invalid image shape. Please provide an image with shape [H, W, C] or [1, H, W, C]."
            )
            print("Provided image shape:", image.shape)
            return (
                jsonify(
                    {
                        "error": "Invalid image shape. Please provide an image with shape [H, W, C] or [1, H, W, C]."
                    }
                ),
                400,
            )

        print("Image array shape:", image.shape)  # Debug: Check input shape

        # Predict using the model
        prediction = model.predict(image)
        predicted_mask = tf.argmax(prediction, axis=-1)
        # predicted_mask = predicted_mask[0]

        color_mask = mask_to_image(predicted_mask[0])
        # color_mask = np.array(class_colors[predicted_mask.numpy().astype(np.int32)])
        # print("Color mask shape:", color_mask.shape)

        unique, counts = np.unique(
            color_mask.reshape(-1, 4), axis=0, return_counts=True
        )
        # unique, counts = np.unique(color_mask.numpy(), return_counts=True)
        print("Unique classes predicted:", unique)
        print("Counts per class:", counts)

        # Convert the color mask to a PIL Image and then to base64 for transmission
        mask_image = Image.fromarray(color_mask, "RGBA")
        buffered = BytesIO()
        mask_image.save(buffered, format="PNG")
        encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({"image": encoded_img})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
