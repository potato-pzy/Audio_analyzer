import os
import joblib

# Define the model path
MODEL_PATH = "C:\\Users\\LENOVO\\Desktop\\project\\model.joblib"

try:
    # Check if the model path exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"The specified model file does not exist: {MODEL_PATH}")

    # Attempt to load the model
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")

except FileNotFoundError as fnf_error:
    print(f"FileNotFoundError: {fnf_error}")
except EOFError:
    print("EOFError: The file might be corrupted or incomplete.")
except ImportError as imp_error:
    print(f"ImportError: {imp_error}. Check for dependency issues.")
except Exception as ex:
    print(f"An unexpected error occurred: {ex}")
