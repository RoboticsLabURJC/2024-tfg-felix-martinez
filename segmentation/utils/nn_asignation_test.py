import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

# 📌 1. Generar datos con puntos bien separados
np.random.seed(42)

# Puntos etiquetados en regiones separadas
puntos_etiquetados = np.array([
    [2, 5, 2], [8, 2, 2], [2, 8, 2], [8, 8, 2],  # Esquinas bajas
    [2, 2, 8], [8, 2, 8], [2, 8, 8], [8, 8, 8],  # Esquinas altas
    [5, 5, 5] # Centro y un extremo
])
etiquetas = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3])  # 5 clases de etiquetas (0-4)

# Puntos sin etiquetas distribuidos aleatoriamente
puntos_sin_etiqueta = np.random.rand(10000, 3) * 10  

# 📌 2. Construcción de KDTree y asignación de etiquetas con vecino más cercano
tree = KDTree(puntos_etiquetados)
distancias, indices = tree.query(puntos_sin_etiqueta)
etiquetas_asignadas = etiquetas[indices]

# 📌 3. Crear nube de puntos para Open3D
puntos_total = np.vstack((puntos_etiquetados, puntos_sin_etiqueta))
colores = np.array([
    [1, 0, 0],  # Rojo
    [0, 1, 0],  # Verde
    [0, 0, 1],  # Azul
    [1, 1, 0],  # Amarillo
    [1, 0, 1]   # Magenta
])

# Colores para la nube completa
colores_total = np.zeros((puntos_total.shape[0], 3))

# Asignar colores a los puntos etiquetados
colores_total[:9] = colores[etiquetas]

# Asignar colores a los puntos sin etiqueta según la etiqueta asignada
colores_total[9:] = colores[etiquetas_asignadas]

# Crear la nube de puntos en Open3D
nube_pcd = o3d.geometry.PointCloud()
nube_pcd.points = o3d.utility.Vector3dVector(puntos_total)
nube_pcd.colors = o3d.utility.Vector3dVector(colores_total)

# 📌 4. Crear esferas grandes para los puntos etiquetados
esferas = []
radio_esfera = 0.4  # Tamaño de las esferas para destacar los puntos etiquetados

for i in range(len(puntos_etiquetados)):
    esfera = o3d.geometry.TriangleMesh.create_sphere(radius=radio_esfera)
    esfera.translate(puntos_etiquetados[i])
    esfera.paint_uniform_color(colores[etiquetas[i]])  # Pintar cada esfera con el color de la etiqueta
    esferas.append(esfera)

# 📌 5. Visualizar en Open3D
import gc
gc.collect()
o3d.visualization.draw_geometries([nube_pcd] + esferas, window_name="Nube con Esferas")
