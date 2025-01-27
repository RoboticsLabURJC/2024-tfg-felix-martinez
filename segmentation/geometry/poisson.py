import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def cargar_nube_bin(archivo_bin):
    nube_puntos = np.fromfile(archivo_bin, dtype=np.float32)
    nube_puntos = nube_puntos.reshape((-1, 4))
    return nube_puntos[:, :3]

# Cargar los datos de la nube de puntos
data = cargar_nube_bin("/home/felix/Escritorio/TFG/datasets/Goose/goose_3d_val/lidar/val/2023-05-17_neubiberg_sunny/2023-05-17_neubiberg_sunny__0396_1684329802563847692_vls128.bin")

# Crear la nube de puntos de Open3D
nube = o3d.geometry.PointCloud()
nube.points = o3d.utility.Vector3dVector(data)

# Filtrar los puntos por encima del suelo (por ejemplo, z > -1.5)
puntos = np.asarray(nube.points)
filtro_suelo = puntos[:, 2] > -1.5  # Ajustar el umbral según el dataset
nube_sin_suelo = nube.select_by_index(np.where(filtro_suelo)[0])

# Verificar si la nube filtrada no está vacía
if nube_sin_suelo.is_empty():
    raise ValueError("La nube de puntos filtrada está vacía. Ajusta el filtro de suelo.")

# Estimar las normales de la nube sin suelo
nube_sin_suelo.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Aplicar la reconstrucción Poisson
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(nube_sin_suelo, depth=9)
print("Malla generada con reconstrucción Poisson para la nube sin suelo.")

# Filtrar la malla usando densidades para eliminar áreas con pocos datos
densidad_media = np.mean(densities)
vertices_a_remover = densities < densidad_media
mesh.remove_vertices_by_mask(vertices_a_remover)

# Visualizar la malla filtrada
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
