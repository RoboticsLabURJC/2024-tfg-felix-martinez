import open3d as o3d
import numpy as np

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
data = cargar_nube_bin("/home/felix/Escritorio/TFG/datasets/Goose/goose_3d_val/lidar/val/2023-05-17_neubiberg_sunny/2023-05-17_neubiberg_sunny__0426_1684329911503929495_vls128.bin")
print(len(data))

# Crear la nube de puntos de Open3D
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(data)

# Crear una esfera centrada en el punto (0, 0, 0)
esfera = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)  # Puedes cambiar el radio si lo deseas
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

