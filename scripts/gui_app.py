import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.model(r'C:\Users\Vishnu R Shetty\Documents\HTML\CSS\PROJECTS\foot_ulcer_detection\foot_ulcer_detection\models')

def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    return 'Foot Ulcer Detected please visit your physician immediately' if prediction[0][0] > 0.5 else 'No foot ulcer detected enjoy your life your healthy '

st.title('Foot Ulcer Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    result = predict_image(img)
    st.write(f'Prediction: {result}')
