import cv2
import time

def detectar_caras():

    # Inicializar variables para medir los FPS
    frames = 0
    start_time = time.time()

    # Cargar el clasificador preentrenado Haar Cascade para detección de caras
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Iniciar el objeto de captura de video utilizando la cámara predeterminada
    cap = cv2.VideoCapture(0)

    # Establecer el tamaño de la ventana de video
    cv2.namedWindow('Detección de caras', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detección de caras', 800, 600)

    while True:
        # Leer un fotograma desde la cámara
        ret, frame = cap.read()

        if not ret:
            break

        # Incrementa el contador de fotogramas
        frames += 1

        # Convertir el fotograma a escala de grises (el clasificador de caras trabaja en escala de grises)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar caras en el fotograma
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Dibujar un rectángulo alrededor de las caras detectadas y agregar texto
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Human face #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        
        # Calcula los FPS y los muestra en el fotograma
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frames / elapsed_time
        fps_str = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Mostrar el fotograma con las caras detectadas
        cv2.imshow('Detección de caras', frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




    # Liberar el objeto de captura y cerrar la ventana
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detectar_caras()
