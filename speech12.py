import os
import time
import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# Create a folder to store audio files
output_folder = 'dataset'
os.makedirs(output_folder, exist_ok=True)

def store_audio_in_file(audio_data, count):
    """Stores audio data into a WAV file."""
    file_path = os.path.join(output_folder, f'audio_{count}.wav')
    with open(file_path, 'wb') as file:
        file.write(audio_data.get_wav_data())

def capture_speech():
    """Captures speech for 5 seconds and returns the audio data."""
    with sr.Microphone() as source:
        print("Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            return audio
        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        return None

if __name__ == "__main__":
    print("Speech capture started. Press Ctrl+C to stop.")
    try:
        count = 1
        while True:
            audio_data = capture_speech()
            if audio_data:
                store_audio_in_file(audio_data, count)
                print(f"Stored audio in audio_{count}.wav")
                count += 1
    except KeyboardInterrupt:
        print("Stopping speech capture...")
    finally:
        print("Audio data saved in folder: dataset")
