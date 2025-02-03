# Add these at the very top of app.py, before any other imports
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress ALL TensorFlow messages
warnings.filterwarnings('ignore')  # Suppress warnings

try:
    import torch
    import numpy as np
    from model import load_model
    import cv2
    import mediapipe as mp
    from flask import Flask, request, jsonify, Response
    from werkzeug.middleware.proxy_fix import ProxyFix
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    from dotenv import load_dotenv
    import threading
    from datetime import datetime
except ImportError as e:
    print(f"Error importing required packages: {str(e)}")
    print("Please make sure all required packages are installed")
    raise

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Setup rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Configuration
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.getenv('MODEL_PATH', 'fitness_model.pth')
CLASSES = ['squats', 'benchpress', 'deadlift']

# Add these global variables
camera = None
output_frame = None
lock = threading.Lock()
rep_counter = 0
exercise_state = False  # False for starting position, True for rep position
last_prediction = None
last_time = datetime.now()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract keypoints from an image
def extract_keypoints(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks]).flatten()
    return keypoints

# Load the model with error handling
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(MODEL_PATH, input_size=99, num_classes=3).to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

def validate_squat(landmarks):
    """Check if pose matches squat form"""
    # Get relevant landmarks
    hip = [landmarks[23].x, landmarks[23].y]
    knee = [landmarks[25].x, landmarks[25].y]
    ankle = [landmarks[27].x, landmarks[27].y]
    
    # Calculate knee angle
    knee_angle = calculate_angle(hip, knee, ankle)
    
    # Squat is valid if knee angle is between 70-100 degrees
    return 70 <= knee_angle <= 100

def validate_deadlift(landmarks):
    """Check if pose matches deadlift form"""
    # Get relevant landmarks
    shoulder = [landmarks[11].x, landmarks[11].y]
    hip = [landmarks[23].x, landmarks[23].y]
    knee = [landmarks[25].x, landmarks[25].y]
    
    # Calculate hip and knee angles
    hip_angle = calculate_angle(shoulder, hip, knee)
    
    # Deadlift is valid if hip angle is between 45-90 degrees
    return 45 <= hip_angle <= 90

def validate_benchpress(landmarks):
    """Check if pose matches bench press form"""
    # Get relevant landmarks
    shoulder = [landmarks[11].x, landmarks[11].y]
    elbow = [landmarks[13].x, landmarks[13].y]
    wrist = [landmarks[15].x, landmarks[15].y]
    
    # Calculate elbow angle
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    
    # Bench press is valid if elbow angle is between 45-90 degrees
    return 45 <= elbow_angle <= 90

# Add this function to track exercise state
def check_exercise_state(landmarks, prediction):
    """Check if the exercise is in starting or rep position"""
    global rep_counter, exercise_state, last_prediction, last_time
    
    current_time = datetime.now()
    time_diff = (current_time - last_time).total_seconds()
    
    # Reset counter if exercise changes
    if last_prediction != prediction:
        rep_counter = 0
        exercise_state = False
        last_prediction = prediction
    
    if prediction == 'squats':
        hip = [landmarks[23].x, landmarks[23].y]
        knee = [landmarks[25].x, landmarks[25].y]
        ankle = [landmarks[27].x, landmarks[27].y]
        knee_angle = calculate_angle(hip, knee, ankle)
        
        # Squat rep logic
        if not exercise_state and knee_angle < 100 and time_diff > 1:
            exercise_state = True
        elif exercise_state and knee_angle > 160 and time_diff > 1:
            exercise_state = False
            rep_counter += 1
            last_time = current_time
            
    elif prediction == 'deadlift':
        shoulder = [landmarks[11].x, landmarks[11].y]
        hip = [landmarks[23].x, landmarks[23].y]
        knee = [landmarks[25].x, landmarks[25].y]
        hip_angle = calculate_angle(shoulder, hip, knee)
        
        # Deadlift rep logic
        if not exercise_state and hip_angle < 60 and time_diff > 1:
            exercise_state = True
        elif exercise_state and hip_angle > 160 and time_diff > 1:
            exercise_state = False
            rep_counter += 1
            last_time = current_time
            
    elif prediction == 'benchpress':
        shoulder = [landmarks[11].x, landmarks[11].y]
        elbow = [landmarks[13].x, landmarks[13].y]
        wrist = [landmarks[15].x, landmarks[15].y]
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        
        # Bench press rep logic
        if not exercise_state and elbow_angle < 90 and time_diff > 1:
            exercise_state = True
        elif exercise_state and elbow_angle > 160 and time_diff > 1:
            exercise_state = False
            rep_counter += 1
            last_time = current_time
    
    return rep_counter

# Modify the generate_frames function to include rep counting
def generate_frames():
    global output_frame, lock
    
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, 
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )
                
                landmarks = results.pose_landmarks.landmark
                pose_valid = False
                
                keypoints = extract_keypoints(frame)
                if keypoints is not None and model is not None:
                    input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        _, predicted = torch.max(outputs, 1)
                        confidence = float(torch.softmax(outputs, 1)[0][predicted].item())
                        prediction = CLASSES[predicted.item()]
                        
                        # Validate pose and count reps
                        if prediction == 'squats' and validate_squat(landmarks):
                            pose_valid = True
                        elif prediction == 'deadlift' and validate_deadlift(landmarks):
                            pose_valid = True
                        elif prediction == 'benchpress' and validate_benchpress(landmarks):
                            pose_valid = True
                        
                        # Count reps if pose is valid
                        reps = check_exercise_state(landmarks, prediction)
                        
                        # Display results
                        if pose_valid and confidence > 0.7:
                            color = (0, 255, 0)
                            status = "CORRECT FORM"
                        else:
                            color = (0, 0, 255)
                            status = "INCORRECT FORM"
                            
                        # Add text to frame
                        cv2.putText(frame, f"{prediction}: {status}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                  color, 2)
                        cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                  color, 2)
                        cv2.putText(frame, f"Reps: {reps}", 
                                  (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                  color, 2)
                        
                        # Add exercise state indicator
                        state_text = "DOWN" if exercise_state else "UP"
                        cv2.putText(frame, f"State: {state_text}", 
                                  (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                  color, 2)
            
            with lock:
                _, buffer = cv2.imencode('.jpg', frame)
                output_frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')
                   
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            continue
    
    cap.release()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
        
    # Check file size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    
    if size > MAX_IMAGE_SIZE:
        return jsonify({'error': 'File too large'}), 400

    try:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        keypoints = extract_keypoints(image)
        if keypoints is None:
            return jsonify({'error': 'No keypoints detected'}), 400

        input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        result = {
            'class': CLASSES[predicted.item()],
            'confidence': float(torch.softmax(outputs, 1)[0][predicted].item())
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera')
def camera():
    return """
    <html>
    <head>
        <title>Fitness Pose Detection</title>
        <style>
            body { 
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                text-align: center;
            }
            img { 
                max-width: 100%;
                height: auto;
                border: 2px solid #333;
                border-radius: 10px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Real-time Fitness Pose Detection</h1>
            <img src="/video_feed">
            <p>Performing real-time pose detection and exercise classification.</p>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("Please use 'python run.py' to start the server")