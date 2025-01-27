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

# Crear una visualización y configurar la cámara
vis = o3d.visualization.Visualizer()
vis.create_window()

vis.add_geometry(nube)

# Configuración de la cámara
ctr = vis.get_view_control()
parameters = ctr.convert_to_pinhole_camera_parameters()

# Modificar la posición de la cámara (vista desde el origen mirando al eje x)
parameters.extrinsic = np.array([
    [1, 0, 0, 0],   # Rotación para que mire al eje x
    [0, 1, 0, 0],   # Mantener la orientación en el eje y
    [0, 0, 1, 0],   # Mantener la orientación en el eje z
    [0, 0, 0, 1]    # La posición de la cámara en (0, 0, 0)
])
ctr.convert_from_pinhole_camera_parameters(parameters)

# Actualizar la visualización
vis.poll_events()
vis.update_renderer()

# Capturar una imagen de profundidad
depth_image = vis.capture_depth_float_buffer(True)
plt.imshow(np.asarray(depth_image), cmap='gray')
plt.title("Simulación de Escaneo - Imagen de Profundidad")
plt.axis('off')
plt.show()

# Liberar la visualización
vis.destroy_window()