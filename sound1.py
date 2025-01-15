import wave
import numpy as np
import matplotlib.pyplot as plt

def display_waveform(audio_file):
    """Displays the waveform of a given WAV audio file."""
    try:
        # Open the WAV file
        with wave.open(audio_file, 'r') as wav_file:
            # Extract audio parameters
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()

            print(f"Channels: {n_channels}")
            print(f"Sample Width: {sampwidth} bytes")
            print(f"Frame Rate: {framerate} Hz")
            print(f"Total Frames: {n_frames}")

            # Read audio frames
            frames = wav_file.readframes(n_frames)

            # Convert to numpy array
            waveform = np.frombuffer(frames, dtype=np.int16)

            # Plot the waveform
            plt.figure(figsize=(10, 4))
            plt.plot(waveform, color='blue')
            plt.title("Waveform of " + audio_file)
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            plt.grid()
            plt.tight_layout()
            plt.show()

    except FileNotFoundError:
        print(f"File not found: {audio_file}")
    except wave.Error as e:
        print(f"Error processing the WAV file: {e}")

if __name__ == "__main__":
    # Set the path of the WAV file inside the code
    audio_path = r"""C:\Users\LENOVO\OneDrive\Desktop\project\dataset\audio_1.wav"""
  # Replace with your file path
    display_waveform(audio_path)
