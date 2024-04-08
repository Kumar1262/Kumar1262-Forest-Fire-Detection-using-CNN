import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("/Users/sagarsharma/Desktop/model.h5")

# Function to predict image label with confidence level
def predict_image_label(img):
    img = image.load_img(img, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input shape
    prediction = model.predict(img_array)
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]  # Calculate confidence level
    label = "Fire" if prediction[0][0] < 0.5 else "No Fire"
    return label, confidence

# Streamlit app
st.title("Forest Fire Detection")

st.markdown("""
<style>
.upload-btn {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        label, confidence = predict_image_label(uploaded_file)
        label_color = 'red' if label == 'Fire' else 'green'
        st.markdown(f"<h1 style='text-align: center; color: black;'>Predicted Label: <span style='color: {label_color}; font-size: 36px;'>{label}</span></h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: black;'>Confidence Level: {confidence:.2f}</h2>", unsafe_allow_html=True)