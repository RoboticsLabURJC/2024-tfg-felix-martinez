import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def cargar_nube_bin(archivo_bin):
    nube_puntos = np.fromfile(archivo_bin, dtype=np.float32)
    nube_puntos = nube_puntos.reshape((-1, 4))
    return nube_puntos[:, :3]

# Cargar los datos de la nube de puntos
data = cargar_nube_bin("/Users/felixmaral/Desktop/TFG/datasets/goose_3d_val/lidar/val/2023-05-15_neubiberg_rain/2023-05-15_neubiberg_rain__0630_1684157871182323213_vls128.bin")

# Crear la nube de puntos de Open3D
nube = o3d.geometry.PointCloud()
nube.points = o3d.utility.Vector3dVector(data)

puntos = np.asarray(nube.points)
filtro_suelo = puntos[:, 2] > -1.85
nube_sin_suelo = nube.select_by_index(np.where(filtro_suelo)[0])

# Aplicar filtro de ruido (outliers estadísticos)
nube_filtrada, ind = nube_sin_suelo.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# Aplicar el algoritmo DBSCAN para segmentación
# epsilon es la distancia máxima entre puntos en un mismo cluster
# min_points es el número mínimo de puntos para formar un cluster
labels = np.array(nube_filtrada.cluster_dbscan(eps=0.3, min_points=30, print_progress=True))

# Asignar colores a los clusters
max_label = labels.max()
print(f"Número de clusters encontrados: {max_label + 1}")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # Puntos ruidosos sin cluster
nube_filtrada.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Crear visualizador
viz = o3d.visualization.Visualizer()
viz.create_window(window_name = "Clustering DBSCAN")

viz.add_geometry(nube_filtrada)
render_op = viz.get_render_option()
render_op.point_size = 1.5

viz.run()
viz.destroy_window()