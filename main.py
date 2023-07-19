import cv2

# Inicializa la captura de video desde la cámara predeterminada
cap = cv2.VideoCapture(0)

while True:
    # Captura un cuadro de video
    ret, frame = cap.read()

    # Muestra el cuadro capturado en una ventana
    cv2.imshow('frame', frame)

    # Espera a que se presione la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera los recursos de la cámara y destruye todas las ventanas abiertas
cap.release()
cv2.destroyAllWindows()
