import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

np.random.seed(42)

puntos_etiquetados = np.array([
    [2, 5, 2], [8, 2, 2], [2, 8, 2], [8, 8, 2], 
    [2, 2, 8], [8, 2, 8], [2, 8, 8], [8, 8, 8],  
    [5, 5, 5] 
])
etiquetas = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3]) 

puntos_sin_etiqueta = np.random.rand(10000, 3) * 10  

tree = KDTree(puntos_etiquetados)
distancias, indices = tree.query(puntos_sin_etiqueta)
etiquetas_asignadas = etiquetas[indices]

puntos_total = np.vstack((puntos_etiquetados, puntos_sin_etiqueta))
colores = np.array([
    [1, 0, 0],  
    [0, 1, 0], 
    [0, 0, 1],  
    [1, 1, 0],  
    [1, 0, 1]   
])

colores_total = np.zeros((puntos_total.shape[0], 3))

colores_total[:9] = colores[etiquetas]

colores_total[9:] = colores[etiquetas_asignadas]

nube_pcd = o3d.geometry.PointCloud()
nube_pcd.points = o3d.utility.Vector3dVector(puntos_total)
nube_pcd.colors = o3d.utility.Vector3dVector(colores_total)

esferas = []
radio_esfera = 0.4  

for i in range(len(puntos_etiquetados)):
    esfera = o3d.geometry.TriangleMesh.create_sphere(radius=radio_esfera)
    esfera.translate(puntos_etiquetados[i])
    esfera.paint_uniform_color(colores[etiquetas[i]]) 
    esferas.append(esfera)

import gc
gc.collect()
o3d.visualization.draw_geometries([nube_pcd] + esferas, window_name="Nube con Esferas")
