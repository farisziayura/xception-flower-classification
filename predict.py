import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

MODEL_PATH = "model/xception_flower.keras"
LABEL_PATH = "model/labels.pkl"

model = load_model(MODEL_PATH)

with open(LABEL_PATH, "rb") as f:
    labels = pickle.load(f)

class_names = {v: k for k, v in labels.items()}


def predict_image(img_path):

    img = image.load_img(
        img_path,
        target_size=(299, 299)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    prediction = model.predict(
        img_array,
        verbose=0
    )

    class_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    flower = class_names[class_index]

    return flower, confidence