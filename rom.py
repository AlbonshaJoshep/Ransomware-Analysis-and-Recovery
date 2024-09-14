import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Function to train and save the model
def train_and_save_model(X_train, y_train, model_filename):
    # Initialize the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Save the model to a file
    with open(model_filename, 'wb') as model_file:
        pickle.dump(clf, model_file)
    print(f"Model saved as '{model_filename}'")

# Function to load and use the model
def load_and_predict(model_filename, X_test):
    # Load the model from the file
    with open(model_filename, 'rb') as model_file:
        clf = pickle.load(model_file)
    
    # Predict using the loaded model
    predictions = clf.predict(X_test)
    return predictions

# Example usage
if _name_ == "_main_":
    # Example data for training
    X_train = np.random.rand(100, 10)  # Replace with actual encrypted data features
    y_train = np.random.randint(0, 2, 100)  # Replace with actual labels
    
    # Train and save the model
    model_filename = 'random_forest_model.pkl'
    train_and_save_model(X_train, y_train, model_filename)
    
    # Example data for prediction
    X_test = np.random.rand(10, 10)  # Replace with actual test data features
    
    # Load the model and make predictions
    predictions = load_and_predict(model_filename, X_test)
    print("Predictions:", predictions)