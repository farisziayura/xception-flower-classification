# 🌸 Flower Classification AI using XCeption

A web-based Flower Classification application built with **Flask** and **Transfer Learning XCeption**. This project can identify flower species from uploaded images using a pre-trained deep learning model.

## 📌 Features

- Flower image classification using XCeption
- Built with Flask
- Upload image through web interface
- Display prediction result and confidence score
- Responsive user interface
- Transfer Learning with TensorFlow/Keras

---

## 🛠️ Technologies Used

- Python 3
- Flask
- TensorFlow / Keras
- XCeption
- NumPy
- OpenCV
- Pillow
- Bootstrap 5
- HTML
- CSS
- JavaScript

---

## 📂 Project Structure

```
xception-flower-classification/
│
├── app.py
├── predict.py
├── train.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── flowers/
│
├── model/
│   ├── labels.pkl
│   └── xception_flower.keras
│
├── static/
│   ├── css/
│   ├── images/
│   ├── js/
│   └── uploads/
│
└── templates/
```

---

## 🌼 Dataset

The model is trained using a flower image dataset containing five classes:

- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

Dataset is divided automatically using **ImageDataGenerator** with:

- Training : 80%
- Validation : 20%

---

## 🧠 Model

Base Model:

- XCeption (ImageNet Pre-trained)

Transfer Learning:

- Freeze base model
- Custom classification layer
- Fine tuning using TensorFlow/Keras

Training Configuration:

- Image Size : 299 × 299
- Epoch : 8
- Batch Size : 32
- Optimizer : Adam
- Loss : Categorical Crossentropy

---

## 📊 Training Result

Training Accuracy:

```
87.71%
```

Best Validation Accuracy:

```
88.60%
```

The best model is automatically saved using **ModelCheckpoint**.

---

## 🚀 Installation

Clone repository

```bash
git clone https://github.com/USERNAME/xception-flower-classification.git
```

Open project

```bash
cd xception-flower-classification
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train Model

```bash
python train.py
```

---

## ▶️ Run Application

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

## 📷 Application Workflow

1. Upload flower image
2. Model processes image
3. XCeption predicts flower class
4. Display prediction and confidence score

---

## 👨‍💻 Developer

**Muhammad Faris**

Universitas Bale Bandung

Program Studi Teknik Informatika

---

## 📄 License

This project is created for academic purposes as part of the Computer Vision course assignment."# xception-flower-classification" 
"# xception-flower-classification" 
