import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
MODEL_PATH = "speaker_recognition_model.pkl"
ALLOWED_EXTENSIONS = {'wav'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define classes (same order as during training)
CLASSES = ["original", "fake"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
    model = None

# Function to predict the speaker of a given audio file
def predict_speaker(file_path):
    try:
        features = extract_features(file_path)
        prediction = model.predict([features])
        predicted_class = CLASSES[prediction[0]]
        return predicted_class
    except Exception as e:
        return f"Error processing file: {str(e)}"

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a WAV file'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first'}), 500
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_speaker(filepath)
        
        # Clean up the uploaded file
        os.remove(filepath)
        
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Create the HTML template
@app.route('/templates/index.html')
def get_template():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Speaker Recognition System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .upload-section {
            margin: 20px 0;
            text-align: center;
        }
        .result {
            margin-top: 20px;
            padding: 10px;
            border-radius: 4px;
            display: none;
        }
        .error {
            background-color: #ffe6e6;
            color: #cc0000;
        }
        .success {
            background-color: #e6ffe6;
            color: #006600;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Speaker Recognition System</h1>
        <div class="upload-section">
            <form id="uploadForm">
                <input type="file" id="audioFile" accept=".wav" required>
                <button type="submit">Analyze</button>
            </form>
        </div>
        <div id="result" class="result"></div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const fileInput = document.getElementById('audioFile');
            const resultDiv = document.getElementById('result');
            
            if (!fileInput.files.length) {
                showResult('Please select a file', true);
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    showResult(`Predicted Speaker: ${data.prediction}`, false);
                } else {
                    showResult(data.error, true);
                }
            } catch (error) {
                showResult('An error occurred while processing the request', true);
            }
        });

        function showResult(message, isError) {
            const resultDiv = document.getElementById('result');
            resultDiv.textContent = message;
            resultDiv.style.display = 'block';
            resultDiv.className = 'result ' + (isError ? 'error' : 'success');
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)