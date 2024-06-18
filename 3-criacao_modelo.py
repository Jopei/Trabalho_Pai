import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Initialize Mediapipe Hands and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the data directory
DATA_DIR = './data'

# Initialize data and labels lists
data = []
labels = []

# Iterate through each subdirectory in the data directory
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue
    
    for img_path in os.listdir(dir_path):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image to detect hands
        results = hands.process(img_rgb)
        
        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                
                # Collect landmark coordinates
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)
                
                # Verify the length of data_aux
                if len(data_aux) != 42:
                    print(f"Erro: data_aux tem comprimento {len(data_aux)}, esperado 42. Pulando esta amostra.")
                    continue
                
                # Append data and label
                data.append(data_aux)
                labels.append(int(dir_))  # Convert directory name to integer
            
# Save the collected data and labels
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
