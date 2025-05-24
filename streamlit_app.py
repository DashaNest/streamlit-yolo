import streamlit as st
import requests
from PIL import Image
import base64
import io

st.title("🎯 YOLOv5 Детектор с FastAPI")

# URL вашего FastAPI сервиса на Амвере
API_URL = "https://fastapi-yolo-darianest.amvera.io"

def image_to_base64(image):
    """Конвертация изображения в base64"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

def call_detection_api(image, confidence):
    """Вызов API для детекции объектов"""
    try:
        image_base64 = image_to_base64(image)
        
        payload = {
            "image": image_base64,
            "confidence": confidence
        }
        
        response = requests.post(
            f"{API_URL}/detect_base64",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка API: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка соединения: {e}")
        return None

def main():
    # Проверка доступности API
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            if health_data.get("model_loaded"):
                st.success("✅ API доступен, модель загружена")
            else:
                st.warning("⚠️ API доступен, но модель не загружена")
        else:
            st.error("❌ API недоступен")
    except:
        st.error("❌ Не удается подключиться к API")
    
    confidence = st.sidebar.slider("Порог уверенности", 0.1, 1.0, 0.5, 0.05)
    
    uploaded_file = st.file_uploader("Загрузите изображение", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Оригинал")
            st.image(image, use_container_width=True)
        
        if st.button("Анализировать изображение"):
            with st.spinner("Обработка..."):
                result = call_detection_api(image, confidence)
                
                if result:
                    with col2:
                        st.subheader("Результат")
                        
                        # Отображаем результат
                        result_image_base64 = result.get("result_image")
                        if result_image_base64:
                            result_image_bytes = base64.b64decode(result_image_base64)
                            result_image = Image.open(io.BytesIO(result_image_bytes))
                            st.image(result_image, use_container_width=True)
                    
                    # Информация об объектах
                    detections = result.get("detections", [])
                    total_objects = result.get("total_objects", 0)
                    
                    if total_objects > 0:
                        st.subheader(f"Найдено объектов: {total_objects}")
                        for detection in detections:
                            st.write(f"• {detection['class_name']}: {detection['confidence']:.2f}")
                    else:
                        st.write("Объекты не найдены")

if __name__ == "__main__":
    main()
