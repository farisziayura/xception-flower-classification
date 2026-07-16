import os
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ======================================================
# PATH
# ======================================================

DATASET_DIR = "dataset/flowers"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# ======================================================
# IMAGE GENERATOR
# ======================================================

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.20,
    rotation_range=20,
    zoom_range=0.20,
    shear_range=0.20,
    horizontal_flip=True,
    fill_mode="nearest"
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(299, 299),
    batch_size=32,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(299, 299),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ======================================================
# SIMPAN LABEL
# ======================================================

labels = train_generator.class_indices

with open(os.path.join(MODEL_DIR, "labels.pkl"), "wb") as f:
    pickle.dump(labels, f)

print("\nClass Label")
print(labels)

# ======================================================
# LOAD XCEPTION
# ======================================================

base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(299, 299, 3)
)

# Freeze semua layer
base_model.trainable = False

# ======================================================
# CUSTOM CLASSIFIER
# ======================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.5)(x)

x = Dense(256, activation="relu")(x)

predictions = Dense(
    train_generator.num_classes,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

# ======================================================
# COMPILE
# ======================================================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================================================
# CALLBACK
# ======================================================

checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "xception_flower.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# ======================================================
# TRAIN
# ======================================================

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=8,
    callbacks=[
        checkpoint,
        earlystop,
        reduce_lr
    ]
)

# ======================================================
# SAVE GRAPH
# ======================================================

plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(MODEL_DIR, "accuracy.png"))
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(MODEL_DIR, "loss.png"))
plt.show()

print("\n===================================")
print("Training selesai")
print("===================================")
print("Model disimpan pada : model/xception_flower.keras")
print("Grafik accuracy : model/accuracy.png")
print("Grafik loss     : model/loss.png")