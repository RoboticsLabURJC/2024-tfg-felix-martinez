import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import open3d as o3d

idx = 0 

# Crear una ventana oculta de Tkinter
root = tk.Tk()
root.withdraw()  # Ocultar la ventana principal

# Ruta inicial
initial_path = Path(r"/home/felix/Escritorio/TFG/datasets/Goose/goose_3d_train")

try:
    # Seleccionar la primera ruta
    print("Selecciona la ruta para 'point_clouds_path'")
    point_clouds_path = Path(filedialog.askdirectory(
        title="Selecciona la carpeta para 'point_clouds_path'",
        initialdir=initial_path
    ))

    # Seleccionar la segunda ruta
    print("Selecciona la ruta para 'labels_path'")
    labels_path = Path(filedialog.askdirectory(
        title="Selecciona la carpeta para 'labels_path'",
        initialdir=initial_path
    ))

    # Mostrar las rutas seleccionadas
    print(f"Ruta seleccionada para point_clouds_path: {point_clouds_path}")
    print(f"Ruta seleccionada para labels_path: {labels_path}")

except Exception as e:
    print(f"Error al seleccionar rutas: {e}")

finally:
    root.destroy()  # Asegúrate de cerrar la ventana de Tkinter

X_files = sorted(os.listdir(point_clouds_path))
Y_files = sorted(os.listdir(labels_path))

# Cargar datos en listas
X = []
Y = []

for x_file, y_file in zip(X_files, Y_files):
    # Leer la nube de puntos
    df_X = pd.read_csv(os.path.join(point_clouds_path, x_file), encoding='ISO-8859-1')
    X.append(df_X)

    # Leer etiquetas
    df_Y = pd.read_csv(os.path.join(labels_path, y_file), encoding='ISO-8859-1')
    Y.append(df_Y)

# Seleccionar una nube de puntos para visualizar

df_points = X[idx]
df_labels = Y[idx]

# Combinar nube de puntos con etiquetas
df_points["sem_label"] = df_labels["sem_label"]

# Normalizar etiquetas para asignar colores
labels = df_points["sem_label"].unique()
num_labels = len(labels)
colors_map = plt.cm.jet(np.linspace(0, 1, num_labels))  # Asignar colores
label_to_color = {label: colors_map[i] for i, label in enumerate(labels)}

# Convertir colores a Open3D (RGB normalizado)
colors = np.array([label_to_color[label] for label in df_points["sem_label"]])[:, :3]  # Solo RGB

# Crear nube de puntos Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(df_points[["x", "y", "z"]].values)
pcd.colors = o3d.utility.Vector3dVector(colors)  # Aplicar colores

# Visualizar con Open3D
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Nube de puntos Open3D")
vis.add_geometry(pcd)

# Ajustar tamaño de los puntos
opt = vis.get_render_option()
opt.point_size = 1 

vis.run()
vis.destroy_window()

