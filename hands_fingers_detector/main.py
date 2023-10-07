import cv2
import mediapipe as mp

# Inicializa el módulo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Inicializa la captura de video desde la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convierte la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta manos en la imagen
    results = hands.process(gray)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Dibuja los puntos de referencia de la mano en la imagen
            for point in landmarks.landmark:
                x, y, _ = map(int, (point.x * frame.shape[1], point.y * frame.shape[0]))
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
