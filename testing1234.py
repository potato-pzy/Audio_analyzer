import os
<<<<<<< HEAD
from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
import joblib
from werkzeug.utils import secure_filename
=======
import uuid
from flask import Flask, request, jsonify, render_template_string,Flask,request,jsonify, render_template_string, render_template, redirect, url_for, session, flash
import numpy as np
import librosa
import joblib
import soundfile as sf
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import scipy.io.wavfile as wav
import requests
from pydub import AudioSegment
import io
>>>>>>> 5c47e85 (added transcription)

app = Flask(__name__)

# Configuration
<<<<<<< HEAD
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
MODEL_PATH = "speaker_recognition_model.pkl"
ALLOWED_EXTENSIONS = {'wav'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define classes (same order as during training)
CLASSES = ["original", "fake"]
=======
UPLOAD_FOLDER = 'files'
MODEL_PATH = "speaker_recognition_model.pkl"
ALLOWED_EXTENSIONS = {'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CLASSES = ["original", "fake"]
# Initialize SQLAlchemy
db = SQLAlchemy()

# Add this model class
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False, default="user")

# Add this decorator function
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('You need to log in first.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Add these configurations in your create_app() or main app
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Change this!
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///speaker_recognition.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db.init_app(app)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# Create all tables
with app.app_context():
    db.create_all()
# HTML template

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speaker Recognition Recorder</title>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            min-height: 100vh;
        }

        .container {
            padding: 16px;
            max-width: 600px;
            margin: 0 auto;
        }

        .card {
            background-color: white;
            border-radius: 16px;
            padding: 24px 16px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 16px;
        }

        h1 {
            color: #1a1a1a;
            font-size: 1.5rem;
            text-align: center;
            margin-bottom: 24px;
        }

        .controls {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-bottom: 24px;
        }

        button {
            background-color: #4CAF50;
            color: white;
            padding: 16px;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
            width: 100%;
            -webkit-tap-highlight-color: transparent;
        }

        button:active {
            transform: scale(0.98);
        }

        button:disabled {
            background-color: #cccccc;
            transform: none;
        }

        .upload-section {
            margin-top: 24px;
            padding-top: 24px;
            border-top: 1px solid #eee;
        }

        .file-input-wrapper {
            margin-bottom: 12px;
        }

        input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 2px dashed #ccc;
            border-radius: 12px;
            margin-bottom: 12px;
        }

        .result {
            margin-top: 20px;
            padding: 16px;
            border-radius: 12px;
            display: none;
            text-align: center;
        }

        .prediction {
            font-weight: 500;
            margin-bottom: 12px;
        }

        .transcription {
            font-size: 1.2rem;
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid #eee;
            line-height: 1.5;
            color: #2c5282;
        }

        .error {
            background-color: #fee2e2;
            color: #dc2626;
        }

        .success {
            background-color: #dcfce7;
            color: #16a34a;
        }

        @media (hover: hover) {
            button:hover {
                background-color: #45a049;
            }
        }

        /* Status indicator */
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
            background-color: #ccc;
        }

        .recording .status-indicator {
            background-color: #dc2626;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% {
                transform: scale(1);
                opacity: 1;
            }
            50% {
                transform: scale(1.2);
                opacity: 0.8;
            }
            100% {
                transform: scale(1);
                opacity: 1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>Speaker Recognition</h1>
            
            <div class="controls">
                <button id="startRecording">
                    <span class="status-indicator"></span>
                    Start Recording
                </button>
                <button id="stopRecording" disabled>Stop Recording</button>
            </div>
            
            <div class="upload-section">
                <form id="uploadForm">
                    <div class="file-input-wrapper">
                        <input type="file" id="audioFile" accept="audio/*">
                    </div>
                    <button type="submit">Upload Audio File</button>
                </form>
            </div>
            
            <div id="result" class="result"></div>
        </div>
    </div>

    <script>
        let mediaRecorder;
        let audioContext;
        let audioChunks = [];
        const startButton = document.getElementById('startRecording');
        const stopButton = document.getElementById('stopRecording');

        async function setupRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        channelCount: 1,
                        sampleRate: 44100
                    }
                });
                
                audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 44100
                });
                
                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm;codecs=opus',
                    audioBitsPerSecond: 128000
                });
                
                mediaRecorder.ondataavailable = async (e) => {
                    audioChunks.push(e.data);
                    if (mediaRecorder.state === "inactive") {
                        const blob = new Blob(audioChunks, { type: 'audio/webm' });
                        await processAndSendAudio(blob);
                    }
                };
                
                startButton.disabled = false;
            } catch (error) {
                console.error('Microphone access denied:', error);
                showResult('Microphone access denied: ' + error.message, true);
            }
        }

        async function processAndSendAudio(blob) {
            try {
                const arrayBuffer = await blob.arrayBuffer();
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                
                // Convert to 16-bit PCM
                const pcmData = new Int16Array(audioBuffer.length);
                const channelData = audioBuffer.getChannelData(0);
                
                for (let i = 0; i < channelData.length; i++) {
                    const s = Math.max(-1, Math.min(1, channelData[i]));
                    pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                
                const wavBlob = new Blob([pcmData.buffer], { type: 'audio/wav' });
                sendData(wavBlob);
            } catch (error) {
                console.error('Error processing audio:', error);
                showResult('Error processing audio: ' + error.message, true);
            }
        }

        document.addEventListener('DOMContentLoaded', setupRecording);

        function sendData(blob) {
            const formData = new FormData();
            formData.append('file', blob, 'recording.wav');
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showResult(data.error, true);
                } else {
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = `
                        <div class="prediction">Predicted Speaker: ${data.prediction}</div>
                        <div class="transcription">${data.transcription}</div>
                    `;
                    resultDiv.style.display = 'block';
                    resultDiv.className = 'result success';
                }
            })
            .catch(error => {
                console.error("Error sending data:", error);
                showResult('Error processing audio: ' + error.message, true);
            });
        }

        function showResult(message, isError) {
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = `<div class="${isError ? '' : 'prediction'}">${message}</div>`;
            resultDiv.style.display = 'block';
            resultDiv.className = 'result ' + (isError ? 'error' : 'success');
        }

        startButton.onclick = e => {
            console.log('Recording started...');
            startButton.disabled = true;
            stopButton.disabled = false;
            startButton.parentElement.classList.add('recording');
            audioChunks = [];
            mediaRecorder.start();
        };

        stopButton.onclick = e => {
            console.log("Recording stopped.");
            startButton.disabled = false;
            stopButton.disabled = true;
            startButton.parentElement.classList.remove('recording');
            mediaRecorder.stop();
        };

        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const fileInput = document.getElementById('audioFile');
            if (!fileInput.files.length) {
                showResult('Please select a file', true);
                return;
            }

            const file = fileInput.files[0];
            try {
                const arrayBuffer = await file.arrayBuffer();
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                
                // Convert to 16-bit PCM
                const pcmData = new Int16Array(audioBuffer.length);
                const channelData = audioBuffer.getChannelData(0);
                
                for (let i = 0; i < channelData.length; i++) {
                    const s = Math.max(-1, Math.min(1, channelData[i]));
                    pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                
                const wavBlob = new Blob([pcmData.buffer], { type: 'audio/wav' });
                sendData(wavBlob);
            } catch (error) {
                showResult('Error processing audio file', true);
            }
        });
    </script>
</body>
</html>
'''
# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
    print(f"Available classes: {CLASSES}")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
    model = None
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
>>>>>>> 5c47e85 (added transcription)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

<<<<<<< HEAD
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
=======
def extract_features(audio_data, sample_rate, n_mfcc=13):
    try:
        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Calculate MFCC features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Concatenate features
        features = np.concatenate((np.mean(mfcc.T, axis=0),
                                 np.mean(delta_mfcc.T, axis=0),
                                 np.mean(delta2_mfcc.T, axis=0)))
        return features
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        raise

def load_and_process_audio(file_path):
    try:
        sample_rate, audio_data = wav.read(file_path)
        
        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
        
        return audio_data, sample_rate
    except Exception as e:
        print(f"Error loading audio: {str(e)}")
        raise

def predict_speaker(file_path):
    try:
        # Load audio using scipy
        audio_data, sample_rate = load_and_process_audio(file_path)
        
        # Extract features
        features = extract_features(audio_data, sample_rate)
        
        # Reshape features for prediction if needed
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        predicted_class = CLASSES[prediction[0]]
        
        print(f"Debug - Prediction made: {predicted_class}")  # Debug line
        return predicted_class
    except Exception as e:
        print(f"Error in predict_speaker: {str(e)}")
        raise

# Add Deepgram transcription function
def transcribe_audio(file_path):
    DEEPGRAM_API_KEY = "5a1b388a7ac22e7d34d3544bae3ab0ab2ea0e6ac"
    URL = "https://api.deepgram.com/v1/listen"
    
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}"
    }
    
    try:
        with open(file_path, 'rb') as audio:
            response = requests.post(
                URL,
                headers=headers,
                data=audio,
                params={
                    "model": "nova-2",
                    "smart_format": "true"
                }
            )
        
        if response.status_code == 200:
            results = response.json()
            return results['results']['channels'][0]['alternatives'][0]['transcript']
        else:
            print(f"Transcription failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        return None

@app.route('/')
@login_required
def index():
    return render_template_string(HTML_TEMPLATE)
>>>>>>> 5c47e85 (added transcription)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
<<<<<<< HEAD
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a WAV file'}), 400
    
=======
>>>>>>> 5c47e85 (added transcription)
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first'}), 500
    
    try:
<<<<<<< HEAD
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
=======
        # Create a temporary WAV file
        wav_output = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + ".wav")
        
        try:
            # Read the audio data directly from the request
            audio_bytes = file.read()
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Normalize the audio
            audio_float = audio_array.astype(np.float32) / np.iinfo(np.int16).max
            
            # Save as WAV using scipy
            wav.write(wav_output, 44100, audio_float)
            
            # Get prediction
            result = predict_speaker(wav_output)
            
            # Get transcription
            transcription = transcribe_audio(wav_output)
            
            response_data = {
                'prediction': result,
                'transcription': transcription if transcription else "Transcription failed"
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return jsonify({'error': 'Error processing audio file'}), 500
        
        finally:
            # Clean up files
            if os.path.exists(wav_output):
                os.remove(wav_output)
            
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add SSL context for HTTPS
if __name__ == "__main__":
    port_number = 5000
    app.run(
        host='0.0.0.0',
        port=port_number,
        debug=True
    )
>>>>>>> 5c47e85 (added transcription)
