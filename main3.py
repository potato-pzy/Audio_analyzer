import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import wave
import librosa
import librosa.display

# Record voice and save to a WAV file
def record_and_save_wav(filename, duration, samplerate=44100):
    print("Recording... Speak now!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished!")

    # Normalize the audio data
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Save as WAV file
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit PC
        wav_file.setframerate(samplerate)
        wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())

    return audio_data, samplerate

# Plot waveform
def plot_waveform(audio_data, samplerate):
    audio_data = audio_data.flatten()  # Ensure 1D
    time_axis = np.linspace(0, len(audio_data) / samplerate, len(audio_data))
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, audio_data, color="blue")
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot spectrogram
def plot_spectrogram(filename):
    audio_data, samplerate = librosa.load(filename, sr=None)
    spectrogram = librosa.stft(audio_data)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=samplerate, x_axis='time', y_axis='hz', cmap='viridis')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

# Main function
if __name__ == "__main__":
    filename = "output.wav"  # File to save the recording
    duration = 5  # Record for 5 seconds

    # Record and save WAV file
    audio_data, samplerate = record_and_save_wav(filename, duration)

    # Plot waveform
    plot_waveform(audio_data, samplerate)

    # Plot spectrogram
    plot_spectrogram(filename)
