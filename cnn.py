import tensorflow as tf
from tensorflow.keras import layers, models
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Hardcoded dataset paths (specify your dataset paths here)
HUMAN_VOICE_DATASET_PATH = "F:\\project\\project\\project\\class A"
ROBOT_VOICE_DATASET_PATH = "F:\\project\\project\\project\\class B"

# Function to augment audio data
def augment_audio(audio, sr):
    # Random time stretch
    stretch_rate = np.random.uniform(0.8, 1.2)
    audio = librosa.effects.time_stretch(audio, rate=stretch_rate)
    
    # Random pitch shift
    pitch_steps = np.random.randint(-4, 4)
    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_steps)
    
    # Add random noise
    noise_amp = 0.005 * np.random.uniform() * np.amax(audio)
    audio = audio + noise_amp * np.random.normal(size=audio.shape[0])
    
    return audio

# Function to extract MFCC features from an audio file
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        audio = augment_audio(audio, sr)  # Apply augmentation
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width < 0:
            mfccs = mfccs[:, :max_pad_len]
        else:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load dataset and preprocess features and labels
def load_dataset(human_path, robot_path):
    features, labels = [], []
    
    # Load human voice dataset
    human_files = [os.path.join(human_path, f) for f in os.listdir(human_path)]
    robot_files = [os.path.join(robot_path, f) for f in os.listdir(robot_path)]
    
    # Balance the dataset
    min_samples = min(len(human_files), len(robot_files))
    human_files = resample(human_files, replace=False, n_samples=min_samples, random_state=42)
    robot_files = resample(robot_files, replace=False, n_samples=min_samples, random_state=42)
    
    for file_path in human_files:
        mfccs = extract_features(file_path)
        if mfccs is not None:
            features.append(mfccs)
            labels.append(0)  # Label 0 for human voice
    
    for file_path in robot_files:
        mfccs = extract_features(file_path)
        if mfccs is not None:
            features.append(mfccs)
            labels.append(1)  # Label 1 for robot voice
    
    return np.array(features), np.array(labels)

# Build a CNN model for binary classification
def build_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

# Train the model on the dataset
def train_model(human_path, robot_path, model_save_path="voice_detector.h5", pkl_save_path="voice_detector.pkl"):
    features, labels = load_dataset(human_path, robot_path)
    features = np.expand_dims(features, axis=-1)  # Add channel dimension
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    model = build_model(features.shape[1:])
    
    # Add callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr])
    
    # Save the model in .h5 format
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save the model in .pkl format using joblib
    joblib.dump(model, pkl_save_path)
    print(f"Model saved to {pkl_save_path}")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    evaluate_model(model, X_val, y_val)

# Plot training and validation accuracy/loss
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Evaluate the model
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_pred = np.round(y_pred).astype(int)
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['Human', 'Robot']))
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_val, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Predict whether an audio file is human or robot voice
def predict_audio(model, file_path):
    mfccs = extract_features(file_path)
    if mfccs is None:
        print("Failed to process the audio file.")
        return
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
    mfccs = np.expand_dims(mfccs, axis=-1)  # Add channel dimension
    prediction = model.predict(mfccs)[0][0]
    if prediction > 0.5:
        print(f"The audio is ROBOTIC with a probability of {prediction:.2f}")
    else:
        print(f"The audio is HUMAN with a probability of {1 - prediction:.2f}")

# Main function to handle training or prediction
def main():
    # Specify the dataset paths directly in the code
    human_path = HUMAN_VOICE_DATASET_PATH
    robot_path = ROBOT_VOICE_DATASET_PATH
    
    # Train the model
    print("Training the model...")
    train_model(human_path, robot_path, model_save_path="voice_detector.h5", pkl_save_path="voice_detector.pkl")
    
    # Example prediction (optional, uncomment to test)
    # model = tf.keras.models.load_model("voice_detector.h5")
    # predict_audio(model, "path_to_your_test_audio_file.wav")

if __name__ == "__main__":
    main()