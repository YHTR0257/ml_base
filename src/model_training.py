# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from src.models import Model  # Assuming a Model class is defined in models.py
from config.train_config import get_train_config  # Assuming a function to get training config

def load_data(file_path):
    """Load dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def train_model(X_train, y_train):
    """Train the model using the training data."""
    model = Model()  # Initialize the model
    model.fit(X_train, y_train)  # Fit the model
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test data."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

def main():
    # Load configuration
    config = get_train_config()
    
    # Load data
    data = load_data(config['data_path'])
    X = data.drop('target', axis=1)  # Assuming 'target' is the label column
    y = data['target']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=config['random_state'])
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Model accuracy: {accuracy:.2f}')
    
    # Save the model
    joblib.dump(model, config['model_save_path'])

if __name__ == "__main__":
    main()