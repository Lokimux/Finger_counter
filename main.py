import cv2
import mediapipe as mp

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize Mediapipe drawing module
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks):
    """
    Function to count the number of extended fingers
    """
    # Tip IDs of the fingers
    finger_tips_ids = [4, 8, 12, 16, 20]

    fingers = []

    # Thumb
    if hand_landmarks.landmark[finger_tips_ids[0]].x < hand_landmarks.landmark[finger_tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other four fingers
    for tip_id in finger_tips_ids[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect the hand
    result = hands.process(rgb_frame)

    # Draw hand landmarks on the frame and count fingers
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count the number of extended fingers
            finger_count = count_fingers(hand_landmarks)

            # Display the finger count
            cv2.putText(frame, f'Fingers: {finger_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Finger Counter', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit on pressing 'Esc'
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
