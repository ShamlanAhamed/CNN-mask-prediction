# image_processing.py
from tensorflow.keras.preprocessing import image
#from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
