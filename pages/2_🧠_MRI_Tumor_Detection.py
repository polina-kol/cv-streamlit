import streamlit as st
from utils.brain_utils import classify_slice, detect_tumor
from PIL import Image

st.title("🧠 Срезы мозга и детекция опухоли")

tab1, tab2 = st.tabs(["Сервис", "Информация о модели"])

with tab1:
    uploaded_files = st.file_uploader("Загрузите MRI срезы", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    for file in uploaded_files:
        image = Image.open(file)
        slice_type = classify_slice(image)
        detection_image = detect_tumor(image, slice_type)
        st.image(detection_image, caption=f"{slice_type.upper()} срез — результат YOLO", use_column_width=True)

with tab2:
    st.header("Классификация срезов (EfficientNet-B0)")
    st.markdown("""
    - **Модель классификации**: EfficientNet-B0
    - **Эпох**: 15
    - **Точность классификации**: 94%

    ---  
    **YOLO модели для детекции опухоли**  
    - 3 отдельных YOLOv5s моделей (по типу среза)
    - **mAP по срезам**: Axial 0.81, Sagittal 0.78, Coronal 0.80  
    """)
    st.image("assets/confusion_matrix_brain.png", caption="Confusion Matrix")

