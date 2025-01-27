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

app = Flask(__name__)

# Configuration
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
            font-weight: 500;
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
                        <input type="file" id="audioFile" accept=".wav">
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

        navigator
            .mediaDevices
            .getUserMedia({audio: true})
            .then(stream => { handlerFunction(stream) })
            .catch(error => {
                showResult('Microphone access denied: ' + error.message, true);
            });

        function handlerFunction(stream) {
            rec = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            rec.ondataavailable = e => {
                audioChunks.push(e.data);
                if (rec.state == "inactive") {
                    convertToWav();
                }
            }
        }

        function convertToWav() {
            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            const fileReader = new FileReader();
            fileReader.onload = function(e) {
                const arrayBuffer = e.target.result;
                audioContext.decodeAudioData(arrayBuffer)
                    .then(function(audioBuffer) {
                        const wavBuffer = audioBufferToWav(audioBuffer);
                        const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
                        sendData(wavBlob);
                    });
            };
            fileReader.readAsArrayBuffer(blob);
        }

        function audioBufferToWav(buffer) {
            const numOfChan = buffer.numberOfChannels,
                  length = buffer.length * numOfChan * 2,
                  buffer_length = buffer.length,
                  sampleRate = buffer.sampleRate;
            
            const wav = new ArrayBuffer(44 + length);
            const view = new DataView(wav);

            // Write WAV header
            writeString(view, 0, 'RIFF');
            view.setUint32(4, 36 + length, true);
            writeString(view, 8, 'WAVE');
            writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, numOfChan, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2 * numOfChan, true);
            view.setUint16(32, numOfChan * 2, true);
            view.setUint16(34, 16, true);
            writeString(view, 36, 'data');
            view.setUint32(40, length, true);

            // Write audio data
            const channelData = [];
            for (let channel = 0; channel < numOfChan; channel++) {
                channelData[channel] = buffer.getChannelData(channel);
            }

            let offset = 44;
            for (let i = 0; i < buffer_length; i++) {
                for (let channel = 0; channel < numOfChan; channel++) {
                    const sample = channelData[channel][i];
                    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
                    offset += 2;
                }
            }

            return wav;
        }

        function writeString(view, offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }

        function sendData(data) {
            var form = new FormData();
            form.append('file', data, 'recording.wav');
            
            $.ajax({
                type: 'POST',
                url: '/predict',
                data: form,
                cache: false,
                processData: false,
                contentType: false
            }).done(function(data) {
                showResult(`Predicted Speaker: ${data.prediction}`, false);
            }).fail(function(jqXHR, textStatus, errorThrown) {
                showResult('Error processing audio: ' + errorThrown, true);
            });
        }

        function showResult(message, isError) {
            const resultDiv = document.getElementById('result');
            resultDiv.textContent = message;
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
    </script>
</body>
</html>
'''
# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
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
        
        # Make prediction
        prediction = model.predict([features])
        predicted_class = CLASSES[prediction[0]]
        return predicted_class
    except Exception as e:
        print(f"Error in predict_speaker: {str(e)}")
        return f"Error processing file: {str(e)}"

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
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a WAV file'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first'}), 500
    
    try:
        file_name = str(uuid.uuid4()) + ".wav"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        file.save(filepath)
        
        result = predict_speaker(filepath)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({'prediction': result})
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port_number = 5000
    app.run(host='0.0.0.0', port=port_number)
