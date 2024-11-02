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
otros_cloud.paint_uniform_color([0, 0, 1])

# Configuración de visualización: ajustar el tamaño de los puntos
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Segmentación del Suelo")
vis.add_geometry(suelo_cloud)
vis.add_geometry(otros_cloud)

# Modificar el tamaño de los puntos
render_option = vis.get_render_option()
render_option.point_size = 1.5  # Ajusta el tamaño de los puntos aquí

# Ejecutar la visualización
vis.run()
vis.destroy_window()

