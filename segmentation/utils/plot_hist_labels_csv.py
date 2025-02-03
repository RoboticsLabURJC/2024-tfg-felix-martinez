import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
import re
import matplotlib.pyplot as plt
import random as rand
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

### ------- Rutas -------- ###

# Crear una ventana oculta de Tkinter
root = tk.Tk()
root.withdraw()  # Ocultar la ventana principal

# Ruta inicial
initial_path = Path(r"/home/felix/Escritorio/TFG")

try:
    # Seleccionar la segunda ruta
    print("Selecciona la ruta para 'labels_path'")
    labels_path = Path(filedialog.askdirectory(
        title="Selecciona la carpeta para 'labels_path'",
        initialdir=initial_path
    ))

    print(f"Ruta seleccionada para labels_path: {labels_path}")

except Exception as e:
    print(f"Error al seleccionar rutas: {e}")

finally:
    root.destroy()

### ------- Cargar CSV -------- ###

csv_files = [file for file in os.listdir(labels_path) if file.endswith('.csv')]

dataframes = []
for file in csv_files:
    filepath = os.path.join(labels_path, file)
    df = pd.read_csv(filepath, header=None, names=['sem_label'])  # Asignar nombre 'sem_label'
    dataframes.append(df)

# Concatenar todos los DataFrames
df_concatenado = pd.concat(dataframes, ignore_index=True)

# Convertir la columna 'sem_label' a numérica
df_concatenado['sem_label'] = pd.to_numeric(df_concatenado['sem_label'], errors='coerce')

# Eliminar valores NaN generados durante la conversión
df_concatenado = df_concatenado.dropna(subset=['sem_label'])

### ------- Visualización ------- ###

from matplotlib.ticker import FuncFormatter

# Crear el histograma
plt.figure(figsize=(16, 6))
#plt.style.use('ggplot')
plt.hist(df_concatenado['sem_label'], 
         bins=np.arange(df_concatenado['sem_label'].min(), df_concatenado['sem_label'].max() + 2), 
         edgecolor='k', 
         alpha=1, 
         align='mid')

# Chatgpt code
# Configurar el formateador para el eje Y
#ax = plt.gca()
#formatter = FuncFormatter(lambda x, _: f"{int(x):,}")  # Formato con separadores de miles
#ax.yaxis.set_major_formatter(formatter)  # Aplicar el formateador

# Personalización del gráfico
plt.title("Histograma de Etiquetas Semánticas", pad=20)
plt.xlabel("Etiqueta Semántica", labelpad=10)
plt.ylabel("Frecuencia", labelpad=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Configurar las marcas del eje X en incrementos de 1
x_ticks = np.arange(df_concatenado['sem_label'].min(), df_concatenado['sem_label'].max() + 2, 1)
plt.xticks(x_ticks, fontsize=8)

# Ajustar padding de los x_ticks
plt.gca().tick_params(axis='x', pad=10)  # Aumenta el espacio entre los ticks y el eje

# Mostrar el gráfico
plt.show()

