# streamlit_app.py
import streamlit as st
from PIL import Image
from model.mask_model import load_mask_model
from utils.image_processing import preprocess_image

model = load_mask_model('mask_find2.keras')

def display_prediction(prediction):
    st.write(f"Prediction: {prediction[0, 0]}")

    if prediction[0, 0] > 0.5:
        st.write('Without Mask')
    else:
        st.write('With Mask')

def main():
    st.title('Your Streamlit App')

    upload_option = st.radio("Choose an option:", ["Image Upload", "Camera"])

    if upload_option == "Image Upload":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img_array = preprocess_image(uploaded_file)

            prediction = model.predict(img_array, batch_size=None, steps=1)
            display_prediction(prediction)

            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    elif upload_option == "Camera":
        picture = st.camera_input("Take a picture")

        if picture is not None:
            img_array = preprocess_image(picture)

            prediction = model.predict(img_array)
            display_prediction(prediction)

if __name__ == "__main__":
    main()
