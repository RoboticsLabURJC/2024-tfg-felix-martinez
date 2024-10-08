import open3d as o3d
import numpy as np
import asyncio
import matplotlib.pyplot as plt
from lector_lidar import leer_archivos_lidar

# Carpeta de los archivos LiDAR
carpeta_lidar = '/home/felix/Escritorio/TFG/datasets/Goose/goose_3d_val/lidar/val/2023-05-17_neubiberg_sunny/'

# Función para visualizar una nube de puntos en Open3D con un mapa de colores basado en remisiones
def visualizar_nube_puntos(puntos, remisiones, vis):
    # Verificar si la nube de puntos está vacía
    if len(puntos) == 0:
        print("[Advertencia] La nube de puntos está vacía, omitiendo visualización.")
        return

    # Crear la nube de puntos si tiene datos
    nube_puntos = o3d.geometry.PointCloud()
    nube_puntos.points = o3d.utility.Vector3dVector(puntos)

    # Normalizar las remisiones a un rango de 0 a 1
    remisiones_normalizadas = np.asarray(remisiones)
    remisiones_normalizadas = (remisiones_normalizadas - remisiones_normalizadas.min()) / (remisiones_normalizadas.max() - remisiones_normalizadas.min())

    # Crear un mapa de color lineal usando matplotlib (por ejemplo, 'jet' o cualquier otro mapa)
    cmap = plt.get_cmap('jet')  # Puedes cambiar a cualquier otro mapa de colores de matplotlib
    colores = cmap(remisiones_normalizadas)[:, :3]  # Obtener solo RGB, sin el canal alfa

    # Asignar los colores a la nube de puntos
    nube_puntos.colors = o3d.utility.Vector3dVector(colores)

    # Limpiar geometrías previas
    vis.clear_geometries()

    # Añadir la nueva nube de puntos
    vis.add_geometry(nube_puntos)

    # Añadir la posición del sensor como una esfera roja en el origen
    sensor_pos = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)  # Tamaño de la esfera
    sensor_pos.translate([0, 0, 0])  # Posición del sensor en el origen
    sensor_pos.paint_uniform_color([1, 0, 0])  # Color rojo para el sensor
    vis.add_geometry(sensor_pos)

    # Configurar la cámara para vista de planta y zoom
    ctr = vis.get_view_control()
    ctr.set_zoom(0.2)  # Ajustar el zoom (reduce el valor para acercar la cámara)
    ctr.set_front([0, 0, 1])  # Vista desde arriba (planta)
    ctr.set_lookat([0, 0, 0])  # El punto de interés (centro de la escena)
    ctr.set_up([0, -1, 0])  # Definir el eje "arriba" para la cámara (invertido para vista de planta)

    # Actualizar la visualización
    vis.update_renderer()
    vis.poll_events()

# Función para actualizar la nube de puntos sin restaurar la cámara
def actualizar_nube_de_puntos(vis, puntos, remisiones):
    # Actualizar la nube de puntos sin intentar restaurar el estado de la cámara
    visualizar_nube_puntos(puntos, remisiones, vis)

# Función principal
async def main():
    # Leer los archivos LiDAR de la carpeta
    datos_lidar = await leer_archivos_lidar(carpeta_lidar)

    # Crear una ventana de visualización Open3D
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()  # Sin fijar el tamaño de la ventana

    # Crear una geometría de nube de puntos vacía
    nube_puntos = o3d.geometry.PointCloud()
    vis.add_geometry(nube_puntos)

    # Obtener las opciones de renderizado para ajustar el tamaño de los puntos
    render_options = vis.get_render_option()
    render_options.point_size = 1.5  # Tamaño pequeño para los puntos
    render_options.light_on = False  # Desactivar la iluminación (útil para nubes de puntos)

    # Índice actual y máximo
    index_actual = [0]  # Usamos una lista mutable para poder modificarlo dentro de los callbacks
    max_index = len(datos_lidar) - 1

    # Mostrar la primera nube de puntos
    puntos, remisiones = datos_lidar[index_actual[0]]
    visualizar_nube_puntos(puntos, remisiones, vis)

    # Callback para avanzar (tecla flecha derecha)
    def avanzar(vis):
        if index_actual[0] < max_index:
            index_actual[0] += 1
            puntos, remisiones = datos_lidar[index_actual[0]]
            actualizar_nube_de_puntos(vis, puntos, remisiones)
            print(f"Mostrando archivo {index_actual[0] + 1}/{max_index + 1}")
        return False

    # Callback para retroceder (tecla flecha izquierda)
    def retroceder(vis):
        if index_actual[0] > 0:
            index_actual[0] -= 1
            puntos, remisiones = datos_lidar[index_actual[0]]
            actualizar_nube_de_puntos(vis, puntos, remisiones)
            print(f"Mostrando archivo {index_actual[0] + 1}/{max_index + 1}")
        return False

    # Asignar los callbacks de las teclas
    vis.register_key_callback(262, avanzar)  # Tecla flecha derecha (código ASCII: 262)
    vis.register_key_callback(263, retroceder)  # Tecla flecha izquierda (código ASCII: 263)

    # Iniciar el loop de visualización
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    asyncio.run(main())