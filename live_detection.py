import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model


MODEL_PATH = 'traffic_sign_classifier_final_model_CLEAN.h5'

LABEL_FILE = 'labels.csv' 
IMG_SIZE = (32, 32)


try:
    data = pd.read_csv(LABEL_FILE)
    sign_names = data['Name'].tolist()
except FileNotFoundError:
    print(f"⚠️ Ошибка: Файл {LABEL_FILE} не найден. Используются ID классов.")
    sign_names = [f"Class {i}" for i in range(58)] 


try:
    model = load_model(MODEL_PATH)
    print(f"✅ Модель загружена: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Ошибка при загрузке модели: {e}")
    exit()


def preprocess_for_prediction(img):
    """Конвертирует в серый, ресайзит до 32x32, нормализует и добавляет размерность."""
    

    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 


    img = cv2.resize(img, IMG_SIZE)
    

    img = img / 255.0 
    

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1) 
    
    return img


cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print(" Невозможно открыть камеру. Проверьте подключение или ID.")
    exit()

print("\n Запуск детектора. Нажмите 'q' для выхода.")

while True:
    success, img = cap.read()
    if not success:
        continue

    
    img_processed = preprocess_for_prediction(img.copy()) 

    
    predictions = model.predict(img_processed, verbose=0)
    class_index = np.argmax(predictions)
    probability = np.max(predictions)

   
    
    
    if probability > 0.70:
        sign_label = sign_names[class_index]
        text = f"{sign_label} ({probability:.2f}%)"
        color = (0, 255, 0) 
    else:
        text = "Searching..."
        color = (0, 0, 255) 
    
    
    cv2.putText(img, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    
    cv2.imshow("Traffic Sign Detector", img)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()