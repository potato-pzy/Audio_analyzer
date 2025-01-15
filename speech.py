import speech_recognition as sr

def speech_to_text():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Adjusting for background noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("You can now speak. I'm listening...")
        
        try:
            # Listen to the audio
            audio = recognizer.listen(source, timeout=5)
            # Recognize speech using Google Web Speech API
            text = recognizer.recognize_google(audio)
            print("You said:", text)

            # Save the recognized text to a file
            with open("recognized_text.txt", "w") as file:
                file.write(text)
            print("Text saved to 'recognized_text.txt'")

        except sr.UnknownValueError:
            print("Sorry, I couldn't understand what you said. Please try again.")
        except sr.RequestError as e:
            print(f"Error with the speech recognition service: {e}")
            print("There was a problem with the service. Please try again later.")

if __name__ == "__main__":
    while True:
        print("\n--- Speech-to-Text Program ---")
        speech_to_text()
        repeat = input("\nWould you like to try again? (yes/no): ").strip().lower()
        if repeat != 'yes':
            print("Exiting the program. Goodbye!")
            break
