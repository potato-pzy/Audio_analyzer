import os
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Function to extract Mel spectrograms from audio files
def extract_mel_spectrogram(audio_file, n_mels=128, duration=2):
    y, sr = librosa.load(audio_file, duration=duration)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

# Function to load data and labels
def load_data_and_labels(data_path, n_mels=128, duration=2):
    X, y = [], []
    classes = os.listdir(data_path)
    for i, cls in enumerate(classes):
        class_path = os.path.join(data_path, cls)
        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)
            mel_spectrogram = extract_mel_spectrogram(file_path, n_mels, duration)
            X.append(mel_spectrogram)
            y.append(i)  # Use class index as label
    return np.array(X), np.array(y)

# Load data and labels
data_path = "C:/Users/LENOVO/Desktop/project/voice analyse/"  # Update with your actual path
X, y = load_data_and_labels(data_path)

# Shuffle the data
X, y = shuffle(X, y, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train = X_train[..., np.newaxis]  # Add channel dimension
X_test = X_test[..., np.newaxis]

# Convert labels to one-hot encoding
num_classes = len(np.unique(y))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save(r'C:\Users\LENOVO\Desktop\project\audio_classification_model.h5')
