import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Function to extract pose keypoints from an image
def extract_keypoints(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks]).flatten()
    return keypoints

# Function to preprocess all images in a folder
def preprocess_data(data_dir):
    X, y = [], []
    classes = os.listdir(data_dir)
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            keypoints = extract_keypoints(image)
            if keypoints is not None:
                X.append(keypoints)
                y.append(class_name)
    return np.array(X), np.array(y)

# Preprocess training and testing data
if __name__ == '__main__':
    train_dir = 'data/train'
    test_dir = 'data/test'

    print("Preprocessing training data...")
    X_train, y_train = preprocess_data(train_dir)

    print("Preprocessing testing data...")
    X_test, y_test = preprocess_data(test_dir)

    # Save preprocessed data as NumPy arrays
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    print("Data preprocessing complete!")