import open3d as o3d
import numpy as np

# Función para cargar el archivo .bin de la nube de puntos
def cargar_nube_bin(archivo_bin):
    nube_puntos = np.fromfile(archivo_bin, dtype=np.float32)
    nube_puntos = nube_puntos.reshape((-1, 4))
    return nube_puntos[:, :3]

# Cargar la nube de puntos
nube_puntos = cargar_nube_bin("/home/felix/Escritorio/TFG/datasets/Goose/goose_3d_val/lidar/val/2022-09-21_garching_uebungsplatz_2/2022-09-21_garching_uebungsplatz_2__0000_1663755178980462982_vls128.bin")

# Crear la nube de puntos en Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(nube_puntos)

# Filtrar por altura (opcional)
# Filtramos para incluir solo puntos con valores z bajos (asumiendo que el suelo es más bajo) RANSAC
z_limite = -1.5  # Ajusta este valor según la altura esperada del suelo en tu dataset
puntos_suelo = np.asarray(pcd.points)
puntos_suelo = puntos_suelo[puntos_suelo[:, 2] < z_limite]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(puntos_suelo)

# Segmentar el plano del suelo
plane_model, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plano del suelo: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# Extraer el plano del suelo
suelo_cloud = pcd.select_by_index(inliers)

# Extraer los puntos restantes
otros_cloud = pcd.select_by_index(inliers, invert=True)

# Colorear el plano del suelo y otros puntos para visualización
suelo_cloud.paint_uniform_color([1, 0, 0])
otros_cloud.paint_uniform_color([1, 1, 1])

# Visualizar el resultado
o3d.visualization.draw_geometries([suelo_cloud, otros_cloud], window_name="Segmentación del Suelo")
