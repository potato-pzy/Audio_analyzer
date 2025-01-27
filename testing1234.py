import os
import uuid
from flask import Flask, request, jsonify, render_template_string
import numpy as np
import librosa
import joblib
import soundfile as sf
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

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Speaker Recognition Recorder</title>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
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
        .controls {
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
            margin: 5px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .upload-section {
            margin: 20px 0;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Speaker Recognition System</h1>
        
        <div class="controls">
            <button id="startRecording">Start Recording</button>
            <button id="stopRecording" disabled>Stop Recording</button>
        </div>
        
        <div class="upload-section">
            <form id="uploadForm">
                <input type="file" id="audioFile" accept=".wav">
                <button type="submit">Upload Audio File</button>
            </form>
        </div>
        
        <div id="result" class="result"></div>
    </div>

    <script>
        let rec;
        let audioChunks = [];

        navigator
            .mediaDevices
            .getUserMedia({audio: true})
            .then(stream => { handlerFunction(stream) });

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

        startRecording.onclick = e => {
            console.log('Recording started...');
            startRecording.disabled = true;
            stopRecording.disabled = false;
            audioChunks = [];
            rec.start();
        };

        stopRecording.onclick = e => {
            console.log("Recording stopped.");
            startRecording.disabled = false;
            stopRecording.disabled = true;
            rec.stop();
        };

        // File upload handling
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
