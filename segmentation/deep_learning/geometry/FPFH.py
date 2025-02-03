# Importar las bibliotecas necesarias
import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Función para cargar una nube de puntos desde un archivo binario
def cargar_nube_bin(archivo_bin):
    nube_puntos = np.fromfile(archivo_bin, dtype=np.float32)
    nube_puntos = nube_puntos.reshape((-1, 4))
    return nube_puntos[:, :3]

# Cargar los datos de la nube de puntos
data = cargar_nube_bin("/Users/felixmaral/Desktop/TFG/datasets/goose_3d_val/lidar/val/2023-05-15_neubiberg_rain/2023-05-15_neubiberg_rain__0630_1684157871182323213_vls128.bin")

# Crear la nube de puntos de Open3D
nube = o3d.geometry.PointCloud()
nube.points = o3d.utility.Vector3dVector(data)

# Verificar que la nube de puntos no esté vacía
if nube.is_empty():
    raise ValueError("La nube de puntos está vacía. Verifica el archivo o la ruta proporcionada.")

# Estimar las normales de la nube de puntos
nube.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Calcular el descriptor FPFH
fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    nube,
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100)
)

# Mostrar información sobre los descriptores
print(f"Descriptor FPFH calculado: {fpfh.data.shape[1]} descriptores de {fpfh.data.shape[0]} dimensiones cada uno")

# Supongamos que ya tienes el objeto `fpfh` calculado
# Convertir los descriptores FPFH a un DataFrame de pandas
fpfh_data = np.asarray(fpfh.data).T  # Transponer para que cada fila sea un descriptor
df_fpfh = pd.DataFrame(fpfh_data)

# Mostrar los primeros 20 descriptores
print("Primeros 20 descriptores FPFH:")
print(df_fpfh.head(20))

# Convertir los descriptores a un array de numpy para el agrupamiento
fpfh_data = np.asarray(fpfh_data.data).T  # Transponer para que cada fila sea un descriptor

# Aplicar DBSCAN para segmentar los puntos basados en descriptores FPFH
dbscan = DBSCAN(eps=0.5, min_samples=10).fit(fpfh_data)
labels = dbscan.labels_

# Agregar los labels a la nube de puntos
colors = plt.get_cmap("tab20")(labels / (labels.max() if labels.max() > 0 else 1))
colors[labels < 0] = 0  # Puntos ruidosos sin cluster
nube.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Visualizar la nube segmentada
o3d.visualization.draw_geometries([nube])