#!/usr/bin/env python3
"""
ASL Sign Classification Prediction Script using Hugging Face Hub (MCP) and Keras with JAX backend.

Usage:
    python predict_hf.py --image path/to/image.jpg [--model hf://JhonArleyCastilloV/ASL_model_1]

Example:
    python predict_hf.py --image test_sign.jpg --model hf://JhonArleyCastilloV/ASL_model_1
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import keras
from huggingface_hub import login

# Set Keras backend to JAX
os.environ["KERAS_BACKEND"] = "jax"

# Optionally, login if you need access to private models
# login(token="TU_TOKEN_HF")

def preprocess_image(image_path, target_size=(224, 224)):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None, None
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size, Image.LANCZOS)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
        img = img.filter(ImageFilter.SHARPEN)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None

def predict(model, img_array):
    try:
        predictions = model(img_array, training=False).numpy()
        classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        classes.extend(["nothing", "del", "space"])
        class_idx = np.argmax(predictions[0])
        prob = predictions[0][class_idx]
        class_name = classes[class_idx] if class_idx < len(classes) else f"Clase {class_idx}"
        top_indices = np.argsort(predictions[0])[::-1][:3]
        top_preds = [(classes[i] if i < len(classes) else f"Clase {i}", float(predictions[0][i])) for i in top_indices]
        return class_name, prob, top_preds
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description='Predicción de señas ASL usando modelo Hugging Face')
    parser.add_argument('--image', '-i', required=True, help='Ruta a la imagen a predecir')
    parser.add_argument('--model', '-m', default='hf://JhonArleyCastilloV/ASL_model_1', help='Ruta MCP al modelo en Hugging Face')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: No se encontró la imagen en {args.image}")
        return

    print(f"Cargando modelo desde: {args.model}")
    try:
        model = keras.saving.load_model(args.model)
        print("Modelo cargado correctamente.")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return

    img_array, img_pil = preprocess_image(args.image)
    if img_array is None:
        return

    class_name, prob, top_preds = predict(model, img_array)
    if class_name is None:
        return

    print("\nResultado de la predicción:")
    print(f"Clase predicha: {class_name}")
    print(f"Probabilidad: {prob:.2%}")
    print("\nTop 3 predicciones:")
    for i, (clase, p) in enumerate(top_preds, 1):
        print(f"{i}. {clase}: {p:.2%}")

if __name__ == "__main__":
    main()
