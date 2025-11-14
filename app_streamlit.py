import streamlit as st
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_stream, VideoProcessorBase, WebRtcMode

# --- КОНСТАНТЫ ---
MODEL_PATH = 'traffic_sign_classifier_final_model_CLEAN.h5'
LABEL_FILE = 'labels.csv' 
IMG_SIZE = (32, 32)
THRESHOLD = 0.70 # Минимальная уверенность для отображения знака

# --- 1. ЗАГРУЗКА МОДЕЛИ И МЕТОК (КЭШИРУЕТСЯ) ---
@st.cache_resource
def load_resources():
    """Загружает модель и метки, кэшируя их."""
    try:
        model = load_model(MODEL_PATH)
        st.success("✅ Модель успешно загружена!")
    except Exception as e:
        # Критическое сообщение, если модель не найдена
        st.error(f"❌ Ошибка при загрузке модели: Убедитесь, что файл '{MODEL_PATH}' находится в корне проекта. Ошибка: {e}")
        return None, None

    try:
        data = pd.read_csv(LABEL_FILE)
        sign_names = data['Name'].tolist()
    except FileNotFoundError:
        st.warning(f"⚠️ Файл {LABEL_FILE} не найден. Используются ID классов.")
        sign_names = [f"Class {i}" for i in range(58)]
        
    return model, sign_names

# --- 2. ФУНКЦИЯ ПРЕОБРАБОТКИ (ДЛЯ МОДЕЛИ) ---
def preprocess_for_prediction(img):
    """Конвертирует в серый, ресайзит до 32x32, нормализует и добавляет размерность."""
    
    # Конвертация в серый цвет, если изображение цветное
    if img.ndim == 3 and img.shape[-1] == 3:
        # WebRTC кадры приходят в BGR, но cvtColor работает как надо
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0 
    
    # Добавление размерности для Keras: (1, 32, 32, 1)
    img = np.expand_dims(img, axis=0) 
    img = np.expand_dims(img, axis=-1)
    return img

# --- 3. КЛАСС ОБРАБОТКИ ВИДЕОПОТОКА ---
class TrafficSignProcessor(VideoProcessorBase):
    def __init__(self, model, sign_names):
        self.model = model
        self.sign_names = sign_names
        
    def recv(self, frame):
        # Преобразование кадра из WebRTC в массив numpy (BGR)
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Предобработка
        img_processed = preprocess_for_prediction(img.copy()) 
        
        # 2. Предсказание
        predictions = self.model.predict(img_processed, verbose=0)
        class_index = np.argmax(predictions)
        probability = np.max(predictions)

        # 3. Визуализация
        if probability > THRESHOLD:
            sign_label = self.sign_names[class_index]
            # Умножаем на 100 для красивого вывода в процентах
            text = f"{sign_label} ({probability*100:.2f}%)" 
            color = (0, 255, 0) # BGR: Зеленый
        else:
            text = "Searching..."
            color = (0, 0, 255) # BGR: Красный
        
        # Наложение текста на кадр OpenCV
        cv2.putText(img, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Возвращаем обработанный кадр обратно в Streamlit
        return img


# --- 4. ОСНОВНОЕ ПРИЛОЖЕНИЕ STREAMLIT ---

def main():
    st.set_page_config(page_title="Traffic Sign Detector", layout="wide")
    st.title("🚗 Детектор Дорожных Знаков (CNN) в Web")
    st.markdown("Проект по распознаванию 58 классов дорожных знаков.")

    # Загружаем ресурсы
    model, sign_names = load_resources()

    if model is None:
        st.stop() # Останавливаем приложение, если модель не загружена

    st.header("Видеопоток с Камеры")
    
    # Запуск WebRTC стрима
    webrtc_stream(
        key="traffic-sign-detector",
        mode=WebRtcMode.SENDRECV,
        # Фабрика создает новый процессор для каждого пользователя
        video_processor_factory=lambda: TrafficSignProcessor(model, sign_names),
        async_processing=True,
        media_stream_constraints={"video": True, "audio": False},
        # Отображение информации о статусе
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    st.markdown("---")
    st.info(f"Модель обучена на 58 классах, достигнута точность: **92.21%**.")

if __name__ == "__main__":
    main()