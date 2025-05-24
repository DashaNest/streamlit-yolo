import streamlit as st
import requests
from PIL import Image
import base64
import io

st.set_page_config(
    page_title="YOLOv5 Детектор",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 YOLOv5 Детектор с FastAPI")

# URL вашего FastAPI сервиса на Амвере
API_URL = "https://fastapi-yolo-darianest.amvera.io"

def image_to_base64(image):
    """Конвертация изображения в base64"""
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        st.error(f"Ошибка конвертации изображения: {e}")
        return None

def call_detection_api(image, confidence):
    """Вызов API для детекции объектов"""
    try:
        image_base64 = image_to_base64(image)
        if not image_base64:
            return None
        
        payload = {
            "image": image_base64,
            "confidence": confidence
        }
        
        response = requests.post(
            f"{API_URL}/detect_base64",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка API: {response.status_code}")
            st.error(f"Детали: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("⏱️ Превышено время ожидания ответа от API")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Ошибка подключения к API")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Ошибка соединения: {e}")
        return None
    except Exception as e:
        st.error(f"🐛 Неожиданная ошибка: {e}")
        return None

def check_api_health():
    """Проверка доступности API"""
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            if health_data.get("model_loaded"):
                st.success("✅ API доступен, модель загружена")
                return True
            else:
                st.warning("⚠️ API доступен, но модель не загружена")
                return False
        else:
            st.error(f"❌ API недоступен (код: {health_response.status_code})")
            return False
    except requests.exceptions.Timeout:
        st.error("⏱️ API не отвечает (timeout)")
        return False
    except Exception as e:
        st.error(f"❌ Не удается подключиться к API: {e}")
        return False

def main():
    # Sidebar с настройками
    with st.sidebar:
        st.subheader("🔧 Настройки")
        st.text(f"API URL: {API_URL}")
        
        # Проверка доступности API
        if st.button("🔄 Проверить API"):
            check_api_health()
        
        confidence = st.slider("Порог уверенности", 0.1, 1.0, 0.5, 0.05)
        
        st.markdown("---")
        st.markdown("**Поддерживаемые форматы:**")
        st.markdown("• PNG, JPG, JPEG")
        st.markdown("• Размер до 200MB")

    # Основной интерфейс
    uploaded_file = st.file_uploader(
        "📁 Загрузите изображение", 
        type=['png', 'jpg', 'jpeg'],
        help="Выберите изображение для анализа объектов"
    )
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Оригинал")
                st.image(image, caption="Загруженное изображение")
                st.caption(f"Размер: {image.size[0]}x{image.size[1]} пикселей")
            
            if st.button("🔍 Анализировать изображение", type="primary"):
                with st.spinner("🤖 Обрабатываем изображение..."):
                    result = call_detection_api(image, confidence)
                    
                    if result:
                        with col2:
                            st.subheader("🎯 Результат")
                            
                            # Отображаем результат
                            result_image_base64 = result.get("result_image")
                            if result_image_base64:
                                try:
                                    result_image_bytes = base64.b64decode(result_image_base64)
                                    result_image = Image.open(io.BytesIO(result_image_bytes))
                                    st.image(result_image, caption="Результат детекции")
                                except Exception as e:
                                    st.error(f"Ошибка отображения результата: {e}")
                        
                        # Информация об объектах
                        detections = result.get("detections", [])
                        total_objects = result.get("total_objects", 0)
                        
                        if total_objects > 0:
                            st.subheader(f"🎯 Найдено объектов: {total_objects}")
                            
                            # Создаем таблицу результатов
                            for i, detection in enumerate(detections, 1):
                                col_icon, col_info = st.columns([1, 4])
                                with col_icon:
                                    st.write(f"**{i}.**")
                                with col_info:
                                    confidence_percent = detection['confidence'] * 100
                                    st.write(f"**{detection['class_name']}** - {confidence_percent:.1f}%")
                                    
                                    # Показываем координаты bbox
                                    bbox = detection.get('bbox', {})
                                    if bbox:
                                        st.caption(f"Координаты: ({bbox.get('x1', 0):.0f}, {bbox.get('y1', 0):.0f}) - ({bbox.get('x2', 0):.0f}, {bbox.get('y2', 0):.0f})")
                        else:
                            st.info("🔍 Объекты не найдены. Попробуйте уменьшить порог уверенности.")
                            
                    else:
                        st.error("❌ Не удалось получить результат от API")
                            
        except Exception as e:
            st.error(f"❌ Ошибка обработки изображения: {e}")
    else:
        # Показываем пример использования
        st.info("👆 Загрузите изображение для начала анализа")
        
        # Автоматическая проверка API при загрузке
        with st.expander("ℹ️ Информация об API", expanded=False):
            st.write("Проверяем доступность API...")
            api_status = check_api_health()
            if api_status:
                st.write("API готов к работе! 🚀")
            else:
                st.write("API недоступен или модель не загружена ⚠️")

if __name__ == "__main__":
    main()
