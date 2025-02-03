import subprocess
import sys
import logging
import os
import warnings

# Suppress TensorFlow and mediapipe messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
warnings.filterwarnings('ignore')  # Suppress warnings

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from waitress import serve
except ImportError:
    print("Installing waitress...")
    install_package('waitress')
    from waitress import serve

from app import app

if __name__ == "__main__":
    print("Starting server with Waitress...")
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("Server is running at http://localhost:5000")
    print("Available endpoints:")
    print("- Health check: http://localhost:5000/health")
    print("- Predict: http://localhost:5000/predict (POST)")
    print("- Camera Feed: http://localhost:5000/camera")
    
    # Run server
    serve(app, host="0.0.0.0", port=5000, threads=6) 