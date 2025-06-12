import os
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
import streamlit as st



# Configuration
UPLOAD_FOLDER = 'files'
MODEL_PATH = "speaker_recognition_model.pkl"
ALLOWED_EXTENSIONS = {'wav'}

 
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# Make sure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title("Upload any file")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # Check file size
    uploaded_file.seek(0, os.SEEK_END)
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)

    if file_size > MAX_CONTENT_LENGTH:
        st.error("File too large! Max size is 16MB.")
    else:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved to {file_path}", ok=True)
        

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

 
from sqlalchemy import create_engine

# SQLite database path
db_uri = "sqlite:///speaker_recognition.db"

# Create engine directly
engine = create_engine(db_uri)



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
                        <input type="file" id="audioFile" accept=".wav,.webm">
                    </div>
                    <button type="submit">Upload Audio File</button>
                </form>
            </div>
            
            <div id="result" class="result"></div>
        </div>
    </div>

    <script>
        let rec;
        let audioChunks = [];
        const startButton = document.getElementById('startRecording');
        const stopButton = document.getElementById('stopRecording');

        // Disable start button initially
        startButton.disabled = true;

        async function setupRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        sampleRate: 44100
                    }
                });
                
                console.log('Microphone permission granted');
                rec = new MediaRecorder(stream, {
                    mimeType: 'audio/webm;codecs=opus'
                });
                
                rec.ondataavailable = e => {
                    audioChunks.push(e.data);
                    if (rec.state === "inactive") {
                        const blob = new Blob(audioChunks, { type: 'audio/webm' });
                        convertToWav(blob).then(wavBlob => {
                            sendData(wavBlob);
                        }).catch(error => {
                            showResult('Error converting audio: ' + error.message, true);
                        });
                    }
                };
                
                startButton.disabled = false;
            } catch (error) {
                console.error('Microphone access denied:', error);
                showResult('Microphone access denied: ' + error.message, true);
            }
        }

        // Function to convert audio blob to WAV format
        async function convertToWav(blob) {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const arrayBuffer = await blob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // Create WAV file
            const numberOfChannels = audioBuffer.numberOfChannels;
            const length = audioBuffer.length * numberOfChannels * 2;
            const buffer = new ArrayBuffer(44 + length);
            const view = new DataView(buffer);
            
            // Write WAV header
            writeString(view, 0, 'RIFF');
            view.setUint32(4, 36 + length, true);
            writeString(view, 8, 'WAVE');
            writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, numberOfChannels, true);
            view.setUint32(24, audioBuffer.sampleRate, true);
            view.setUint32(28, audioBuffer.sampleRate * numberOfChannels * 2, true);
            view.setUint16(32, numberOfChannels * 2, true);
            view.setUint16(34, 16, true);
            writeString(view, 36, 'data');
            view.setUint32(40, length, true);

            // Write audio data
            const channels = [];
            for (let i = 0; i < numberOfChannels; i++) {
                channels.push(audioBuffer.getChannelData(i));
            }

            let offset = 44;
            for (let i = 0; i < audioBuffer.length; i++) {
                for (let channel = 0; channel < numberOfChannels; channel++) {
                    const sample = Math.max(-1, Math.min(1, channels[channel][i]));
                    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
                    offset += 2;
                }
            }

            return new Blob([buffer], { type: 'audio/wav' });
        }

        function writeString(view, offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
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
            rec.start();
        };

        stopButton.onclick = e => {
            console.log("Recording stopped.");
            startButton.disabled = false;
            stopButton.disabled = true;
            startButton.parentElement.classList.remove('recording');
            rec.stop();
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
                let blob;
                if (file.type.includes('webm')) {
                    blob = await convertToWav(file);
                } else {
                    blob = file;
                }

                const formData = new FormData();
                formData.append('file', blob, 'audio.wav');

                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (response.ok) {
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = `
                        <div class="prediction">Predicted Speaker: ${data.prediction}</div>
                        <div class="transcription">${data.transcription}</div>
                    `;
                    resultDiv.style.display = 'block';
                    resultDiv.className = 'result success';
                } else {
                    showResult(data.error, true);
                }
            } catch (error) {
                showResult('An error occurred while processing the request', true);
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first'}), 500
    
    try:
        # Save the uploaded file temporarily with .webm extension
        temp_input = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + ".webm")
        wav_output = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + ".wav")
        
        try:
            # Save the uploaded file
            file.save(temp_input)
            
            # Load audio using librosa
            audio_data, sr = librosa.load(temp_input, sr=44100, mono=True)
            
            # Save as WAV using soundfile
            sf.write(wav_output, audio_data, sr, format='WAV')
            
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
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(wav_output):
                os.remove(wav_output)
            
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add SSL context for HTTPS
