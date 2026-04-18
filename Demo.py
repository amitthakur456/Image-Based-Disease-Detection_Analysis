import streamlit as st
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# page configuration
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="centered"
)

# load trained model
@st.cache_resource
def load_trained_model():
    
    model = load_model("pneumonia_model.h5")
    
    return model

model = load_trained_model()

# title
st.title("🫁 Pneumonia Detection from Chest X-ray")

st.write("""
Upload a chest X-ray image to detect Pneumonia using Deep Learning.
Model: DenseNet121 (Transfer Learning)
""")

# upload image
uploaded_file = st.file_uploader(
    "Upload X-ray image",
    type=["jpg","png","jpeg"]
)

# prediction function
def predict_image(img):

    img = img.resize((224,224))

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    probability = float(prediction[0][0])

    return probability


# display result
if uploaded_file:

    img = image.load_img(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            img,
            caption="Uploaded Image",
            use_column_width=True
        )

    with col2:

        st.subheader("Prediction")

        prob = predict_image(img)

        if prob > 0.5:

            st.error("Pneumonia Detected")

            st.write(f"Probability: {prob:.2f}")

        else:

            st.success("Normal")

            st.write(f"Probability: {prob:.2f}")

        st.progress(prob)


# footer
st.write("---")

st.caption("AI Model for educational purpose")