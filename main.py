import cv2
from PIL import Image
from emotional import predict_emotion
from spotify_integration import suggest_music, play_song

# Access the webcam
cap = cv2.VideoCapture(0)

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
        emotion = predict_emotion(image)

        # Suggest music
        music_suggestions = suggest_music(emotion)

        # Display emotion and songs on the frame
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y_offset = 100
        for song in music_suggestions[:3]:  # Display top 3 songs
            text = f"{song['name']} by {song['artist']}"
            cv2.putText(frame, text, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 30

        # Display the video feed
        cv2.imshow("Emotion Detection and Music Suggestions", frame)

    except Exception as e:
        print(f"Error: {e}")
        break

    # Exit on 'q'
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('p'):  # Play the first suggested song on 'p' key
        if music_suggestions:
            play_song(music_suggestions[0]["uri"])

cap.release()
cv2.destroyAllWindows()
