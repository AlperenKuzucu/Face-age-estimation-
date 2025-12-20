import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ====== KONFİG ======
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data" / "raw" / "imdb_crop" # Yolu düzelttim
TARGET_SIZE = (64, 64)      
MAX_IMAGES = 100000          
EPOCHS = 20                 
BATCH_SIZE = 64             

def load_labels_from_imdb_filenames():
    records = []
    for root, _, files in os.walk(DATA_ROOT):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parts = os.path.splitext(fname)[0].split("_")
            if len(parts) < 4: continue
            try:
                birth_year = int(parts[2].split("-")[0])
                photo_year = int(parts[3].split("-")[0])
                age = photo_year - birth_year
            except ValueError: continue
            if age <= 0 or age > 100: continue
            filepath = str(Path(root) / fname)
            records.append({"filepath": filepath, "age": age})
    return pd.DataFrame(records)

def preprocess_image_for_cnn(path, target_size=TARGET_SIZE):
    img = Image.open(path).convert("L")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr  

def build_dataset(max_images=MAX_IMAGES):
    df = load_labels_from_imdb_filenames()
    if len(df) > max_images:
        df = df.sample(n=max_images, random_state=42).reset_index(drop=True)
    X_list, y_list = [], []
    for i, row in df.iterrows():
        try:
            img_arr = preprocess_image_for_cnn(row["filepath"])
            X_list.append(img_arr[..., np.newaxis])
            y_list.append(row["age"])
        except: continue
    return np.array(X_list), np.array(y_list)

def build_cnn_model(input_shape=(64, 64, 1)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="linear")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mae", metrics=["mae"])
    return model

def train_cnn():
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = build_cnn_model(input_shape=X_train.shape[1:])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
    model.save(BASE_DIR / "face_age_cnn.h5")

if __name__ == "__main__":
    train_cnn()
