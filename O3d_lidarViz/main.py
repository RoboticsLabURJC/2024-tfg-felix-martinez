import open3d as o3d
import numpy as np
import asyncio
from lector_lidar import leer_archivos_lidar

# Carpeta de los archivos LiDAR
carpeta_lidar = '/home/felix/Escritorio/TFG/datasets/Goose/goose_3d_val/lidar/val/2023-05-17_neubiberg_sunny/'

# Función para mostrar una nube de puntos en Open3D
def visualizar_nube_puntos(puntos, vis):
    # Actualizar la nube de puntos en la visualización
    nube_puntos = o3d.geometry.PointCloud()
    nube_puntos.points = o3d.utility.Vector3dVector(puntos)

    # Limpiar geometrías previas
    vis.clear_geometries()

    # Añadir la nube de puntos
    vis.add_geometry(nube_puntos)

    # Añadir la posición del sensor como una esfera roja en el origen
    sensor_pos = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)  # Tamaño de la esfera
    sensor_pos.translate([0, 0, 0])  # Posición del sensor en el origen
    sensor_pos.paint_uniform_color([1, 0, 0])  # Color rojo
    vis.add_geometry(sensor_pos)

    # Actualizar la visualización
    vis.update_renderer()
    vis.poll_events()

# Función principal
async def main():
    # Leer los archivos LiDAR de la carpeta
    datos_lidar = await leer_archivos_lidar(carpeta_lidar)

    # Crear una ventana de visualización Open3D
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # Crear una geometría de nube de puntos vacía
    nube_puntos = o3d.geometry.PointCloud()
    vis.add_geometry(nube_puntos)

    # Obtener el control de la cámara (view control)
    ctr = vis.get_view_control()
    
    # Configuración de la cámara
    ctr.set_lookat([0, 0, 0])  # Apuntar al origen
    ctr.set_front([0, 0, -1])  # Dirección de la cámara (hacia el eje -Z)
    ctr.set_up([0, -1, 0])     # Dirección hacia arriba (eje Y negativo)
    ctr.set_zoom(5)          # Aumentar el zoom (más cercano a la escena)

    # Obtener las opciones de renderizado para reducir el tamaño de los puntos
    render_options = vis.get_render_option()
    render_options.point_size = 1.5  # Tamaño pequeño para los puntos del scatter plot

    # Índice actual y máximo
    index_actual = [0]  # Usamos una lista mutable para poder modificarlo dentro de los callbacks
    max_index = len(datos_lidar) - 1

    # Mostrar la primera nube de puntos
    puntos, remisiones = datos_lidar[index_actual[0]]
    visualizar_nube_puntos(puntos, vis)

    # Callback para avanzar (tecla flecha derecha)
    def avanzar(vis):
        if index_actual[0] < max_index:
            index_actual[0] += 1
            puntos, remisiones = datos_lidar[index_actual[0]]
            visualizar_nube_puntos(puntos, vis)
            print(f"Mostrando archivo {index_actual[0] + 1}/{max_index + 1}")
        return False

    # Callback para retroceder (tecla flecha izquierda)
    def retroceder(vis):
        if index_actual[0] > 0:
            index_actual[0] -= 1
            puntos, remisiones = datos_lidar[index_actual[0]]
            visualizar_nube_puntos(puntos, vis)
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
