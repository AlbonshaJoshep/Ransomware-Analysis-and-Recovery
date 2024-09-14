import numpy as np
import pickle
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

# Load the trained model
def load_model(model_filename):
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Function to preprocess image (convert to grayscale, resize, and flatten it)
def preprocess_image(image_path, target_size=(64, 64)):
    image = imread(image_path)
    gray_image = rgb2gray(image)
    resized_image = resize(gray_image, target_size, anti_aliasing=True)
    flat_image = resized_image.flatten()
    return flat_image

# Function to encrypt the image using the trained model
def encrypt_image(image_path, model):
    flat_image = preprocess_image(image_path)  # Process the image
    flat_image = np.reshape(flat_image, (1, -1))  # Reshape for prediction
    encrypted_data = model.predict(flat_image)
    return encrypted_data

# Example usage
if _name_ == "_main_":
    model_filename = 'random_forest_image_model.pkl'
    model = load_model(model_filename)
    
    image_path = '/home/kali/Downloads/cybersecurity.jpg'  # Replace with actual image path
    encrypted_image = encrypt_image(image_path, model)
    print("Encrypted Data:", encrypted_image)