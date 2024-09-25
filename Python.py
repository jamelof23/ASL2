import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3

# Load the trained model
model = load_model('asl_hand_landmarks_model.h5')

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Mapping of class indices to ASL letters and signs
class_names = ['A', 'B', 'C', ..., 'Z', 'delete', 'space', 'nothing']
sentence = []
last_predicted_letter = ''

# Webcam capture
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks and make a prediction
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            landmarks = np.expand_dims(landmarks, axis=0)
            prediction = model.predict(landmarks)
            predicted_class = np.argmax(prediction)

            # Map prediction to ASL character
            predicted_letter = class_names[predicted_class]

            # Speak and write the letter when it's detected
            if predicted_letter != last_predicted_letter and predicted_letter != 'nothing':
                if predicted_letter == 'delete':
                    if sentence:
                        sentence.pop()  # Remove the last letter
                        engine.say("delete")
                        engine.runAndWait()
                elif predicted_letter == 'space':
                    sentence.append(' ')
                    engine.say("space")
                    engine.runAndWait()
                else:
                    sentence.append(predicted_letter)
                    engine.say(predicted_letter)  # Spell the letter
                    engine.runAndWait()

                last_predicted_letter = predicted_letter

            # If 'nothing' is detected, read the full sentence
            if predicted_letter == 'nothing' and sentence:
                full_sentence = "".join(sentence).strip()
                engine.say(f"The sentence is: {full_sentence}")
                engine.runAndWait()

    # Display the sentence on the video frame
    cv2.putText(frame, "Sentence: " + "".join(sentence), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('ASL Sign Language Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
