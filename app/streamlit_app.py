# streamlit_app.py
import streamlit as st
from PIL import Image
from model.mask_model import load_mask_model
from utils.image_processing import preprocess_image

model = load_mask_model('mask_find3.h5')

title_alignment = """
    <style>
        .e1eexb540 {
            text-align: center;
            color: red;
            font-style: italic;
            margin-left: 40px;
        }
        .stAlert{
            margin-right: 40px;
            margin-left: -15px;
        }
    </style>
"""

def load_and_predict(uploaded_file):
    img_array = preprocess_image(uploaded_file)
    prediction = model.predict(img_array, batch_size=None, steps=1)
    return prediction

def display_prediction(prediction):
    if prediction[0, 0] > 0.5:
        st.header('Without Mask')
        st.warning('Please wear mask')
    else:
        st.header('With Mask')
        st.success('Great! go ahead')
        st.balloons()

def main():
    
    with st.sidebar:
        st.markdown("<h1 style='color: black; text-align: center;  margin-top: -30px; margin-bottom:30px; font-weight: bold;'>Why Mask</h1>", unsafe_allow_html=True)
        
        # Add an image to the sidebar
        #st.image("https://png.pngtree.com/png-vector/20220513/ourmid/pngtree-ecological-stop-co2-emissions-sign-on-white-background-png-image_4595665.png", use_column_width=True)
        
        # Add a radio button to select the ML model
        st.markdown("<h3 style='text-align: center; color: black; margin-bottom: -50px; font-weight: bold;'>Advantages of mask:</h3>", unsafe_allow_html=True)
        
    
    #st.set_page_config(layout="centered")
    st.title('Input | Capture faces to detect mask')
    
    col1, col2, col3 = st.columns([1, 1, 1]) 

    with col2:
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center; margin:1px;}</style>', unsafe_allow_html=True)
        st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
        
        upload_option = st.radio("Choose an option:", ["Image Upload", "Camera"])

    if upload_option == "Image Upload":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            prediction = load_and_predict(uploaded_file)
            
            col1, col2, col3 = st.columns([1, 1, 1]) 
            with col2:
                st.markdown(title_alignment, unsafe_allow_html=True)
                display_prediction(prediction)

            col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])
            
            with col2:
                st.image(uploaded_file, caption='Uploaded Image.', width=300)

    elif upload_option == "Camera":
        col1, col2, col3 = st.columns([1, 1, 1])
        picture = st.camera_input("Take a picture")
        
        if picture is not None:
            img_array = preprocess_image(picture)
            prediction = model.predict(img_array)
            
            with col2:
                st.markdown(title_alignment, unsafe_allow_html=True)
                display_prediction(prediction)

if __name__ == "__main__":
    main()
