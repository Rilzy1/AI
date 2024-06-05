import cv2
import tensorflow as tf

# Assuming model is already loaded and trained
model.save('emotion_detection_model.h5')

# Emotion categories (adjust according to your model's training data)
emotion_categories = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for emotion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 48, 48, 1)

    # Predict emotion
    prediction = model.predict(reshaped)
    max_index = prediction[0].argmax()
    emotion = emotion_categories[max_index]

    # Display the emotion on the frame
    cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
