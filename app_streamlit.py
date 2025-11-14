import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras 
from PIL import Image # Используем PIL (Pillow) для чтения изображения (она обычно устанавливается автоматически)

# --- КОНСТАНТЫ ---
MODEL_PATH = 'traffic_sign_classifier_final_model_CLEAN.h5'
LABEL_FILE = 'labels.csv' 
IMG_SIZE = (32, 32)
THRESHOLD = 0.70 # Минимальная уверенность для отображения знака

# --- 1. ЗАГРУЗКА МОДЕЛИ И МЕТОК (КЭШИРУЕТСЯ) ---
@st.cache_resource
def load_resources():
    try:
        model = keras.models.load_model(MODEL_PATH)
        st.success("✅ Модель успешно загружена!")
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке модели: {e}")
        return None, None

    try:
        data = pd.read_csv(LABEL_FILE)
        sign_names = data['Name'].tolist()
    except FileNotFoundError:
        st.warning(f"⚠️ Файл {LABEL_FILE} не найден.")
        sign_names = [f"Class {i}" for i in range(58)]
        
    return model, sign_names

# --- 2. ФУНКЦИЯ ПРЕОБРАБОТКИ (БЕЗ CV2) ---
def preprocess_for_prediction(pil_img):
    """Преобразует изображение PIL для предсказания моделью."""
    # Конвертируем в серый, если нужно
    if pil_img.mode != 'L':
        pil_img = pil_img.convert('L')
        
    # Изменение размера
    pil_img = pil_img.resize(IMG_SIZE)
    
    # Преобразование в массив numpy
    img = np.array(pil_img, dtype=np.float32)
    
    # Нормализация
    img = img / 255.0 
    
    # Keras ожидает форму (1, 32, 32, 1)
    img = np.expand_dims(img, axis=0) 
    img = np.expand_dims(img, axis=-1)
    return img

# --- 3. ОСНОВНОЕ ПРИЛОЖЕНИЕ STREAMLIT ---

def main():
    st.set_page_config(page_title="Traffic Sign Classifier", layout="wide")
    st.title("🚦 Классификатор Дорожных Знаков по Фото")
    st.markdown("Загрузите изображение дорожного знака для классификации.")

    model, sign_names = load_resources()

    if model is None:
        st.stop()

    uploaded_file = st.file_uploader("Выберите изображение...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Чтение файла с помощью PIL (Pillow)
        image = Image.open(uploaded_file)
        
        # 1. Показ загруженного изображения
        st.image(image, caption="Загруженное изображение", use_column_width=True)

        # 2. Предобработка и предсказание
        if st.button("Классифицировать"):
            with st.spinner('Анализ изображения...'):
                img_processed = preprocess_for_prediction(image)
                
                predictions = model.predict(img_processed, verbose=0)
                class_index = np.argmax(predictions)
                probability = np.max(predictions)

                # 3. Вывод результата
                if probability > THRESHOLD:
                    sign_label = sign_names[class_index]
                    st.success(f"✅ **Результат:** {sign_label}")
                    st.metric(label="Уверенность модели", value=f"{probability*100:.2f}%")
                else:
                    st.warning(f"⚠️ **Результат:** Неопределенный знак (Уверенность {probability*100:.2f}%)")
                    st.info("Попробуйте загрузить более четкое изображение.")

if __name__ == "__main__":
    main()

