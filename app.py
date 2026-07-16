import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template(
            "index.html",
            error="Silakan pilih gambar terlebih dahulu."
        )

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "index.html",
            error="Belum ada file yang dipilih."
        )

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="File harus berupa JPG, JPEG, atau PNG."
        )

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    flower, confidence = predict_image(filepath)

    return render_template(
        "result.html",
        image=filename,
        flower=flower.title(),
        confidence=round(confidence * 100, 2)
    )


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)