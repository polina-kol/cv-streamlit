import streamlit as st
from utils.face_utils import detect_and_blur_faces
import requests
import os
from PIL import Image
from io import BytesIO

st.title("😷 Детекция лиц и блюринг")

tab1, tab2 = st.tabs(["Сервис", "Информация о модели"])

with tab1:
    source = st.radio("Источник изображений:", ["Загрузка файлов", "Ссылка (URL)"])
    
    if source == "Загрузка файлов":
        uploaded_files = st.file_uploader("Загрузите изображение(я)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        for file in uploaded_files:
            image = Image.open(file)
            result = detect_and_blur_faces(image)
            st.image(result, caption=f"Обработано: {file.name}", use_column_width=True)
    else:
        url = st.text_input("Введите URL изображения:")
        if st.button("Обработать по ссылке"):
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                result = detect_and_blur_faces(image)
                st.image(result, caption="Обработанное изображение", use_column_width=True)
            except:
                st.error("Ошибка при загрузке изображения.")

with tab2:
    st.header("YOLOv5 face detector + OpenCV Blur")
    st.markdown("""
    - **Модель**: YOLOv5s, дообученная на [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)
    - **Метрика mAP**: 0.89
    - **Эпох**: 50  
    """)
    st.image("assets/pr_curve_face.png", caption="PR-кривая")