import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

# ====== KONFİG ======
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data" / "raw" / "imdb_crop"
TARGET_SIZE = (64, 64)
MAX_IMAGES = 20000 

def load_labels_from_imdb_filenames():
    records = []
    for root, _, files in os.walk(DATA_ROOT):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")): continue
            parts = os.path.splitext(fname)[0].split("_")
            if len(parts) < 4: continue
            try:
                birth_year = int(parts[2].split("-")[0])
                photo_year = int(parts[3].split("-")[0])
                age = photo_year - birth_year
            except: continue
            if 0 < age <= 100:
                records.append({"filepath": str(Path(root) / fname), "age": age})
    return pd.DataFrame(records)

def preprocess_image_flatten(path, target_size=TARGET_SIZE):
    # CNN'den farkı: Resmi düzleştiriyoruz (1 Boyutlu vektör yapıyoruz)
    img = Image.open(path).convert("L")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.flatten()

def train_baseline():
    df = load_labels_from_imdb_filenames()
    if len(df) > MAX_IMAGES:
        df = df.sample(n=MAX_IMAGES, random_state=42)
    
    X_list, y_list = [], []
    for _, row in df.iterrows():
        try:
            X_list.append(preprocess_image_flatten(row["filepath"]))
            y_list.append(row["age"])
        except: continue
        
    X = np.array(X_list)
    y = np.array(y_list)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Random Forest Eğitiliyor...")
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Baseline Model MAE: {mae:.2f}")
    
    joblib.dump(model, BASE_DIR / "face_age_rf.joblib")

if __name__ == "__main__":
    train_baseline()
