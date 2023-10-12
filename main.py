import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import keras
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas

model = load_model(filepath="C:/Users/Jules/Documents/Cours/master5/pythonProject/model")
drawing_mode = "freedraw"
stroke_width = 3
stroke_color = "#fff"
bg_color = "#000"
realtime_update = True

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=None,
    update_streamlit=realtime_update,
    height=255,
    width=255,
    drawing_mode=drawing_mode,
    point_display_radius=0,
    key="canvas",
)

if st.button("voir ma prediction"):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        img = img.convert('L')
        img.save("test.png")
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)
        st.write(f"Ma prédiction est : {predicted_label}")