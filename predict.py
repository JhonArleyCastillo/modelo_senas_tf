#!/usr/bin/env python3
"""
ASL Sign Classification Prediction Script.

This script provides functionality to make predictions using the improved 
ASL sign classification model from the Senas_model.ipynb notebook.

Compatible with TensorFlow/Keras models (.keras, .h5 formats).

Usage:
    python predict.py --image path/to/image.jpg [--model path/to/model.keras]

Example:
    python predict.py --image test_sign.jpg --visualize --model modelo_senas.keras
"""

import os
import argparse
from typing import Tuple, Dict, Any, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def cargar_modelo(ruta_modelo: str) -> Optional[tf.keras.Model]:
    """
    Load a TensorFlow/Keras model from the specified path.
    
    Args:
        ruta_modelo (str): Path to the model file (.keras or .h5).
        
    Returns:
        Optional[tf.keras.Model]: Loaded model if successful, None otherwise.
        
    Raises:
        FileNotFoundError: If model file doesn't exist.
        ValueError: If model format is not supported.
    """
    if not os.path.exists(ruta_modelo):
        print(f"Error: Model file not found at {ruta_modelo}")
        return None
        
    try:
        print(f"Loading model from: {ruta_modelo}")
        modelo = load_model(ruta_modelo)
        print(f"Model loaded successfully: {modelo.name}")
        return modelo
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocesar_imagen(
    ruta_imagen: str, 
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[Optional[np.ndarray], Optional[Image.Image]]:
    """
    Preprocess an image for model prediction with quality enhancements.
    
    Args:
        ruta_imagen (str): Path to the input image.
        target_size (Tuple[int, int]): Target size for resizing (width, height).
        
    Returns:
        Tuple[Optional[np.ndarray], Optional[Image.Image]]: 
            Preprocessed image array and PIL image, or (None, None) if error.
            
    Raises:
        FileNotFoundError: If image file doesn't exist.
        ValueError: If image cannot be processed.
    """
    if not os.path.exists(ruta_imagen):
        print(f"Error: Image file not found at {ruta_imagen}")
        return None, None
        
    try:
        # Load image with PIL for better preprocessing options
        img = Image.open(ruta_imagen).convert("RGB")
        
        # Resize to target dimensions using high-quality resampling
        img = img.resize(target_size, Image.LANCZOS)
        
        # Enhance image quality for better recognition
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)  # Increase contrast by 30%
        
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)  # Increase brightness by 10%
        
        # Apply sharpening filter to improve edge definition
        img = img.filter(ImageFilter.SHARPEN)
        
        # Convert to numpy array and normalize pixel values to [0, 1]
        img_array = np.array(img) / 255.0
        
        # Add batch dimension for model input
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None
        img = enhancer.enhance(1.1)
        
        # Aplicar filtro de nitidez
        img = img.filter(ImageFilter.SHARPEN)
        
        # Convertir a array y normalizar
        img_array = np.array(img) / 255.0
        
        # Expandir dimensiones para batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    except Exception as e:
        print(f"Error en el preprocesamiento: {e}")
        return None, None

def predecir(modelo, img_array):
    """
    Realiza la predicción con el modelo.
    
    Args:
        modelo: El modelo cargado
        img_array: Array numpy de la imagen preprocesada
        
    Returns:
        Clase predicha, probabilidad y todas las predicciones
    """
    try:
        # Predicción
        predicciones = modelo.predict(img_array, verbose=0)
        
        # Clases disponibles
        clases = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        clases.extend(["nothing", "del", "space"])
        
        # Obtener la clase con mayor probabilidad
        clase_idx = np.argmax(predicciones[0])
        probabilidad = predicciones[0][clase_idx]
        
        # Nombre de la clase (si está dentro del rango)
        if clase_idx < len(clases):
            nombre_clase = clases[clase_idx]
        else:
            nombre_clase = f"Clase {clase_idx}"
        
        # Ordenar todas las predicciones
        indices_ordenados = np.argsort(predicciones[0])[::-1]
        todas_predicciones = {
            clases[i] if i < len(clases) else f"Clase {i}": float(predicciones[0][i])
            for i in indices_ordenados[:5]  # Top 5
        }
        
        return nombre_clase, probabilidad, todas_predicciones
    except Exception as e:
        print(f"Error en la predicción: {e}")
        return None, None, None

def visualizar_resultado(img_pil, nombre_clase, probabilidad, todas_predicciones):
    """
    Visualiza el resultado de la predicción.
    
    Args:
        img_pil: Imagen PIL
        nombre_clase: Nombre de la clase predicha
        probabilidad: Probabilidad de la predicción
        todas_predicciones: Diccionario con todas las predicciones
    """
    try:
        # Crear una figura con 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Mostrar la imagen
        ax1.imshow(img_pil)
        ax1.set_title(f'Predicción: {nombre_clase} ({probabilidad:.2%})', fontsize=14)
        ax1.axis('off')
        
        # Gráfico de probabilidades para top-5
        clases = list(todas_predicciones.keys())
        probs = list(todas_predicciones.values())
        
        y_pos = np.arange(len(clases))
        ax2.barh(y_pos, [prob * 100 for prob in probs], color=sns.color_palette("viridis", len(probs)))
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(clases)
        ax2.set_xlabel('Probabilidad (%)')
        ax2.set_title('Top predicciones', fontsize=14)
        
        # Añadir etiquetas de porcentaje
        for i, v in enumerate(probs):
            ax2.text(v * 100 + 1, i, f'{v:.2%}', va='center')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error en la visualización: {e}")

def main():
    """Función principal"""
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Predicción de señas ASL')
    parser.add_argument('--image', '-i', required=True, help='Ruta a la imagen a predecir')
    parser.add_argument('--model', '-m', default='modelo_senas_final.keras', 
                        help='Ruta al modelo entrenado (.keras o .h5)')
    parser.add_argument('--visualize', '-v', action='store_true', 
                        help='Mostrar visualización del resultado')
    args = parser.parse_args()
    
    # Verificar que la imagen existe
    if not os.path.exists(args.image):
        print(f"Error: No se encontró la imagen en {args.image}")
        return
    
    # Cargar modelo
    modelo = cargar_modelo(args.model)
    if modelo is None:
        return
    
    # Preprocesar imagen
    img_array, img_pil = preprocesar_imagen(args.image)
    if img_array is None:
        return
    
    # Realizar predicción
    nombre_clase, probabilidad, todas_predicciones = predecir(modelo, img_array)
    if nombre_clase is None:
        return
    
    # Mostrar resultado
    print("\nResultado de la predicción:")
    print(f"Clase predicha: {nombre_clase}")
    print(f"Probabilidad: {probabilidad:.2%}")
    
    # Mostrar top 3 predicciones
    print("\nTop 3 predicciones:")
    i = 1
    for clase, prob in list(todas_predicciones.items())[:3]:
        print(f"{i}. {clase}: {prob:.2%}")
        i += 1
    
    # Visualizar si se solicitó
    if args.visualize:
        visualizar_resultado(img_pil, nombre_clase, probabilidad, todas_predicciones)

if __name__ == "__main__":
    main()
