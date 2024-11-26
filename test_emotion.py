import cv2
from PIL import Image  # Make sure to import Image from PIL
from emotional import predict_emotion  # Assuming the emotion prediction function is in emotional.py

# Access the webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, or try 1 if it's not working

# Check if the webcam was successfully opened
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the webcam.")
        break

    try:
        # Convert the frame to a PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Predict emotion
        predicted_emotion = predict_emotion(image)  # Get predicted emotion

        # Show the predicted emotion on the frame
        cv2.putText(frame, f"Emotion: {predicted_emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with prediction
        cv2.imshow("Emotion Detection", frame)

    except Exception as e:
        print(f"Error during emotion detection: {e}")
        break

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
