import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model and class names
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('waste_model.keras')
    return model

model = load_model()

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

st.title("♻️ Smart Waste Classification System")
st.write("Upload an image of waste material to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]) if predictions.shape[-1] > 1 else predictions[0]
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    st.subheader("Class Probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[0][i]*100:.2f}%")
        st.progress(float(predictions[0][i]))