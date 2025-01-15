import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Path to save the model
MODEL_PATH = "speaker_recognition_model.pkl"

# Define classes (same order as during training)
CLASSES = ["original", "fake"]

# Function to extract MFCC and delta features from audio
def extract_features(file_path, n_mfcc=13):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    # Concatenate MFCC, delta, and delta-delta features
    features = np.concatenate((np.mean(mfcc.T, axis=0),
                               np.mean(delta_mfcc.T, axis=0),
                               np.mean(delta2_mfcc.T, axis=0)))
    return features

# Function to load the dataset
def load_data(dataset_path, classes):
    features, labels = [], []
    for label, class_name in enumerate(classes):
        class_folder = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_folder):
            continue
        for file_name in os.listdir(class_folder):
            if file_name.endswith(".wav"):
                file_path = os.path.join(class_folder, file_name)
                try:
                    mfcc_features = extract_features(file_path)
                    features.append(mfcc_features)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
    return np.array(features), np.array(labels)

# Function to train the model
def train_model(dataset_path, classes):
    print("Loading dataset...")
    X, y = load_data(dataset_path, classes)
    print(f"Dataset loaded with {len(X)} samples.")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")

    # Save the trained model to file
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Path to the dataset folder
DATASET_PATH = "voice analyse"  # Path to the folder containing person1, person2, etc.

# Train the model
train_model(DATASET_PATH, CLASSES)
