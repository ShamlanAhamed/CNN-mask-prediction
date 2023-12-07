from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf


model = load_model('mask_find2.keras')

# Streamlit app
st.title('Your Streamlit App')

# Choose between image upload and camera option
upload_option = st.radio("Choose an option:", ["Image Upload", "Camera"])

if upload_option == "Image Upload":
    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the image
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((64, 64))  # Resize the image to match your model's expected sizing
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize the pixel values to be between 0 and 1
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array, batch_size=None, steps=1)  # gives all class probabilities

        # Display the prediction
        st.write(f"Prediction: {prediction[0, 0]}")

        # Display text based on prediction
        if prediction[0, 0] > 0.5:
            st.write('Without Mask')
        else:
            st.write('With Mask')

        # Optionally, you can display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

elif upload_option == "Camera":
    # Take a picture using the camera
    picture = st.camera_input("Take a picture")

    if picture is not None:
        # Convert the picture to an image
        img = Image.open(picture)
        img = img.resize((64, 64))  # Resize the image to match your model's expected sizing
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize the pixel values to be between 0 and 1
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        prediction = model.predict(img_array)

        # Display the prediction
        st.write(f"Prediction: {prediction[0, 0]}")

        # Display text based on prediction
        if prediction[0, 0] > 0.5:
            st.write('Without Mask')
        else:
            st.write('With Mask')
