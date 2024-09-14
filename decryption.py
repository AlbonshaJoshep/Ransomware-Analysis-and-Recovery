import numpy as np
import pickle

# Load the trained model
def load_model(model_filename):
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Mapping from model output to actual images or data
def get_decrypted_content(prediction):
    # Define your mapping here
    mapping = {
        0: '/home/kali/Downloads/cs.jpeg',
        1: '/home/kali/Downloads/cybersecurity.jpg'
    }
    return mapping.get(prediction, 'Unknown')

# Decrypt the encrypted data
def decrypt_image(encrypted_data, model):
    # Ensure encrypted_data is in the correct shape
    if encrypted_data.shape[1] != 4096:
        raise ValueError("Encrypted data does not have the correct number of features.")
    
    # Predict (classify) the encrypted data using the trained model
    prediction = model.predict(encrypted_data)
    
    # Map the prediction to the actual content
    decrypted_content = get_decrypted_content(prediction[0])
    return decrypted_content

# Example usage
if _name_ == "_main_":
    model_filename = 'random_forest_image_model.pkl'
    model = load_model(model_filename)
    
    # Example encrypted data (ensure it has 4096 features)
    encrypted_image = np.array([[1] * 4096])  # Example with dummy data
    
    decrypted_content = decrypt_image(encrypted_image, model)
    print("Decrypted Content:", decrypted_content)