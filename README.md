title: ASL Image

emoji: 🐨

colorFrom: blue

colorTo: purple

sdk: gradio

sdk_version: 5.41.0

app_file: app.py

pinned: false

license: mit

short_description: prediccion de señas

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

Clasificador de Lengua de Señas ASL
Este repositorio contiene un modelo de deep learning para el reconocimiento de señas del alfabeto americano (ASL) usando imágenes. El modelo está entrenado con Keras y expuesto mediante una interfaz web usando Gradio.

Archivos incluidos
model.keras: Modelo entrenado en formato Keras (debe subirse a Hugging Face o incluirse aquí).
app.py: Script de inferencia y demo web con Gradio.
requirements.txt: Dependencias necesarias para ejecutar el modelo y la demo.
README.md: Este archivo.
Uso
Puedes probar el modelo localmente ejecutando:

pip install -r requirements.txt
python app.py
Esto abrirá una interfaz web donde puedes subir una imagen de una seña y obtener la predicción.

Cómo funciona
El modelo espera imágenes de tamaño 224x224 píxeles.
El preprocesamiento incluye ajuste de contraste, brillo y nitidez.
El modelo predice una de las 29 clases: las 26 letras del alfabeto, más "nothing", "del" y "space".
Ejemplo de uso en código
from PIL import Image
import keras
import numpy as np

model = keras.saving.load_model("model.keras")
img = Image.open("ruta_a_tu_imagen.jpg").convert("RGB").resize((224, 224))
img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
pred = model(img_array, training=False).numpy()
clases = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["nothing", "del", "space"]
print("Predicción:", clases[np.argmax(pred[0])])
Requisitos
Python 3.8+
TensorFlow/Keras
Gradio
PIL (Pillow)
numpy
Instala las dependencias con:

pip install -r requirements.txt
Créditos
Desarrollado por Jhon Arley Castillo V y colaboradores.

Licencia
MIT
