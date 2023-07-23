# import cv2


# # Inicializa la captura de video desde la cámara predeterminada
# cap = cv2.VideoCapture(0)

# while True:
#     # Captura un cuadro de video
#     ret, frame = cap.read()

#     # Muestra el cuadro capturado en una ventana
#     cv2.imshow('frame', frame)

#     # Espera a que se presione la tecla 'q' para salir del bucle
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Libera los recursos de la cámara y destruye todas las ventanas abiertas
# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Cargar el modelo entrenado
model = keras.models.load_model('modelo.h5')

# Crear un objeto de la clase Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# Crear un objeto de la clase DrawingUtils
mpDraw = mp.solutions.drawing_utils

# Crear un objeto de la clase VideoCapture
cap = cv2.VideoCapture(0)

# Crear un diccionario con las etiquetas de las clases
labels = {0:'Cinco', 1:'Cuatro', 2:'Tres', 3:'Dos', 4:'Uno', 5:'Cero'}

while True:
    # Leer un frame del video
    success, img = cap.read()
    # Convertir la imagen a RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Procesar la imagen con el objeto hands
    results = hands.process(imgRGB)
    # Obtener las coordenadas de las manos detectadas
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Dibujar las marcas y las conexiones de la mano
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # Crear una lista con las coordenadas x e y de cada marca
            coords = []
            for lm in handLms.landmark:
                coords.append([lm.x, lm.y])
            # Convertir la lista en un array de numpy
            coords = np.array(coords)
            # Normalizar las coordenadas entre 0 y 1
            coords = coords / np.max(coords)
            # Redimensionar el array para que tenga la forma esperada por el modelo (1, 42)
            coords = coords.reshape(1, -1)
            # Predecir la clase de la mano usando el modelo
            prediction = model.predict(coords)
            # Obtener el índice de la clase con mayor probabilidad
            index = np.argmax(prediction)
            # Obtener la etiqueta correspondiente al índice
            label = labels[index]
            # Mostrar la etiqueta en la imagen
            cv2.putText(img, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # Mostrar la imagen en una ventana
    cv2.imshow('Image', img)
    # Esperar a que se presione la tecla ESC para salir del bucle
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
