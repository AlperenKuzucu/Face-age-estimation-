import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2  

# train_cnn içindeki fonksiyonları kullanabilmek için
# Bu dosya src klasöründe olduğu için sys.path ayarı gerekebilir
# Ancak basitlik adına fonksiyonu burada tekrar tanımlayabilir veya 
# train_cnn'den import edebilirsin. Şimdilik import kalsın.
try:
    from train_cnn import preprocess_image_for_cnn
except ImportError:
    # Eğer import hatası olursa basit preprocess fonksiyonunu buraya koyuyoruz
    def preprocess_image_for_cnn(path, target_size=(64, 64)):
        img = Image.open(path).convert("L")
        img = img.resize(target_size)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

# ====== PATH AYARLARI ======
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "face_age_cnn.h5"
TEST_DIR = BASE_DIR / "samples"  # GitHub için klasör adını samples yaptık

def load_model_disk():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")
    print(f"Model yükleniyor: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="mae", metrics=["mae"])
    return model

def detect_face_bbox(image_path: str):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if len(faces) == 0:
        return img_rgb, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return img_rgb, (x, y, w, h)

def show_image_with_age(image_path: str, age: float):
    img_rgb, bbox = detect_face_bbox(image_path)
    if img_rgb is None:
        pil_img = Image.open(image_path).convert("RGB")
        img_rgb = np.array(pil_img)
        bbox = None

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.axis("off")  
    if bbox is not None:
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor="blue", facecolor="none")
        ax.add_patch(rect)
    ax.set_title(f"Tahmini yaş: {age:.1f}", fontsize=14)
    plt.tight_layout()
    plt.show()

def predict_single_image(model, image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Resim bulunamadı: {image_path}")
    print(f"Resim işleniyor: {image_path}")
    img_arr = preprocess_image_for_cnn(image_path)  
    img_arr = img_arr[..., np.newaxis]              
    img_arr = np.expand_dims(img_arr, axis=0)       
    pred = model.predict(img_arr, verbose=0)[0, 0]
    print(f"Tahmini yaş: {pred:.2f}")
    show_image_with_age(image_path, pred)
    return pred

def main():
    try:
        model = load_model_disk()
    except Exception as e:
        print(e)
        return

    if len(sys.argv) > 1:
        predict_single_image(model, sys.argv[1])
    else:
        print("Lütfen bir resim yolu belirtin.")

if __name__ == "__main__":
    main()
