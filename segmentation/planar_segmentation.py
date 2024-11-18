import open3d as o3d
import numpy as np
import sys
import os
from plyfile import PlyData

def cargar_nube_bin(archivo_bin):
    nube_puntos = np.fromfile(archivo_bin, dtype=np.float32)
    nube_puntos = nube_puntos.reshape((-1, 4))
    return nube_puntos[:, :3]

def visualize(vis, geometries):
    vis.create_window()
    for geom in geometries:
        vis.add_geometry(geom)  # Agrega cada geometría de forma individual
    opt = vis.get_render_option()
    opt.point_size = 1.6
    vis.run()
    vis.destroy_window()

# Función para calcular el ángulo entre dos vectores
def calcular_angulo(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) * (180.0 / np.pi)

# Cargar los datos de la nube de puntos
#data = cargar_nube_bin("/home/felix/Escritorio/TFG/2024-tfg-felix-martinez/Lidar-Visualizer/data/examples/goose/2023-05-17_neubiberg_sunny__0381_1684329746496937615_vls128.bin")

# Crear la nube de puntos de Open3D
#point_cloud = o3d.geometry.PointCloud()
#point_cloud.points = o3d.utility.Vector3dVector(data)

file_path = "/home/felix/Escritorio/TFG/2024-tfg-felix-martinez/Lidar-Visualizer/data/examples/rellis3d/frame000004-1581797150_855.ply"

if not file_path.endswith('.ply') or not os.path.exists(file_path):
        print(f"Error: .ply file not found at {file_path}")
        sys.exit(1)
plydata = PlyData.read(file_path)
print(file_path)
x, y, z = plydata['vertex'].data['x'], plydata['vertex'].data['y'], plydata['vertex'].data['z']
points = np.vstack((x, y, z)).T

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

point_cloud = point_cloud.voxel_down_sample(voxel_size=0.1)

# Crear una esfera centrada en el punto (0, 0, 0)
esfera = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)  # Puedes cambiar el radio si lo deseas
esfera.translate((0, 0, 0))  # Mover la esfera al punto (0, 0, 0)
# Colorear la esfera (opcional)
esfera.paint_uniform_color([1, 1, 0])  # Color rojo

# Calcular las normales de la nube de puntos
point_cloud.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
)

# Detectar parches planos en la nube de puntos
plane_patches = point_cloud.detect_planar_patches(
    normal_variance_threshold_deg=30.0, # Limite de la variacion permitida entre las normales de los puntos de una region (Grados). Cuanto más pequeño, más estricto
    coplanarity_deg=20, # Establece la distribucion de la distancia entre los puntos que forman el plano. Cuanto mas pequeño, mas estricto. (grados)
    outlier_ratio=1.0, # Porcentaje maximo de puntos rechazados en la región para rechazar el plano estimado
    min_plane_edge_length=0.1, # Distancia minima de la arista mas larga del plano (filtrar planos pequeños)
    min_num_points=0, # Cantidad minima de puntos para que se acepte el plano en la region
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30) # Define el parametro de vecinos mas cercanos para la realizacion de los otros algoritmos
)

vertical_tolerance = 20
# Crear meshes de los parches planos para visualizarlos
meshes = []
for patch in plane_patches:
    # La normal del plano es la tercera columna de la matriz de rotación R del OrientedBoundingBox
    normal = patch.R[:, 2]
    angulo_con_vertical = calcular_angulo(normal, np.array([0, 0, 1]))

    # Incluir solo parches oblicuos o verticales, excluir horizontales con una tolerancia de ±10°
    if angulo_con_vertical > vertical_tolerance and angulo_con_vertical < (180 - vertical_tolerance):
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
            patch, scale=[1, 1, 0.0001]  # Escala para aplanar los parches
        )
        mesh.paint_uniform_color([1, 0, 0])  # Color rojo para los parches
        meshes.append(mesh)

# Visualizar los meshes junto con la nube de puntos
vis = o3d.visualization.Visualizer()
visualize(vis, [point_cloud] + meshes + [esfera])

