import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("face_recognition_model_2.h5")

# Define label mapping (update with your actual label mapping used during training)
label_map = {0: 'ankit', 1: 'not ankit'}  # Update with your label mapping

# Function to preprocess a single frame from camera
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    frame = cv2.resize(frame, (50, 50))  # Resize frame to 50x50 pixels (adjust if necessary)
    frame = frame / 255.0  # Normalize pixel values to be between 0 and 1
    frame = np.expand_dims(frame, axis=-1)  # Add a channel dimension (for grayscale)
    return frame

# Initialize video capture from camera (default camera or external camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    
    if not ret:
        print("Failed to capture frame from camera. Exiting...")
        break
    
    # Preprocess the frame for prediction
    preprocessed_frame = preprocess_frame(frame)
    
    # Reshape the frame for prediction (add an extra dimension for batch)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
    
    # Make prediction
    prediction = model.predict(preprocessed_frame)
    
    # Decode the prediction
    predicted_label = np.argmax(prediction)  # Get the index of the highest probability
    confidence = np.max(prediction)  # Get the confidence (probability) of the prediction
    predicted_person = label_map.get(predicted_label, "Unknown")  # Get the predicted person's name from label_map
    
    # Display predicted label and confidence on the frame
    text = f"Predicted: {predicted_person} (Confidence: {confidence:.2f})"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Face Recognition', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
