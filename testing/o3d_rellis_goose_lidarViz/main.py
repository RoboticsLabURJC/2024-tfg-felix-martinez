import open3d as o3d
import numpy as np
import asyncio
import matplotlib.pyplot as plt
from lector_lidar import leer_archivos_lidar
import argparse

# Colormaps disponibles
colormaps = ['jet', 'viridis', 'plasma', 'inferno']  # Diferentes mapas de colores
colormap_idx = [0]  # Índice mutable para cambiar el mapa

# Estado para vista reducida o completa
mostrar_reducido = [False]

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

    # Crear un mapa de color lineal usando matplotlib basado en el colormap actual
    cmap = plt.get_cmap(colormaps[colormap_idx[0]])  # Mapa de colores actual
    colores = cmap(remisiones_normalizadas)[:, :3]  # Obtener solo RGB, sin el canal alfa

    # Asignar los colores a la nube de puntos
    nube_puntos.colors = o3d.utility.Vector3dVector(colores)

    # Limpiar geometrías previas
    vis.clear_geometries()

    # Añadir la nueva nube de puntos
    vis.add_geometry(nube_puntos)

    # Añadir la posición del sensor como un cilindro rojo en el origen
    sensor_pos = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3, height=1.0)  # Crear un cilindro
    sensor_pos.translate([0, 0, 0])  # Posición del sensor en el origen
    sensor_pos.paint_uniform_color([1, 0, 0])  # Color rojo para el sensor
    vis.add_geometry(sensor_pos)

    # Añadir ejes de coordenadas en el origen para referencia
    ejes_coordenadas = o3d.geometry.TriangleMesh.create_coordinate_frame(size=4.0, origin=[0, 0, 0])  # Tamaño ajustable
    vis.add_geometry(ejes_coordenadas)

    # Configurar la cámara para vista de planta y zoom
    ctr = vis.get_view_control()
    ctr.set_zoom(0.06)  # Ajustar el zoom
    ctr.set_front([0, 0, 1])  # Vista desde arriba (planta)
    ctr.set_lookat([0, 0, 0])  # El punto de interés (centro de la escena)
    ctr.set_up([0, -1, 0])  # Definir el eje "arriba" para la cámara

    # Actualizar la visualización
    vis.update_renderer()
    vis.poll_events()

# Función principal
async def main(carpeta_lidar, tipo_archivo):
    # Validar tipo de archivo
    if tipo_archivo not in ['bin', 'ply']:
        print(f"[ERROR] Tipo de archivo '{tipo_archivo}' no es válido. Debe ser 'bin' o 'ply'.")
        return

    # Leer los archivos LiDAR de la carpeta de forma asincrónica y ordenada
    try:
        datos_lidar = await leer_archivos_lidar(carpeta_lidar, tipo_archivo)
    except Exception as e:
        print(f"[ERROR] Ocurrió un error durante la lectura de archivos: {e}")
        return

    if len(datos_lidar) == 0:
        print(f"[ADVERTENCIA] No se encontraron archivos de tipo '{tipo_archivo}' en la carpeta proporcionada.")
        return

    # Crear una ventana de visualización Open3D
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()  # Sin fijar el tamaño de la ventana

    # Crear una geometría de nube de puntos vacía
    nube_puntos = o3d.geometry.PointCloud()
    vis.add_geometry(nube_puntos)

    # Mostrar la primera nube de puntos
    puntos, remisiones = datos_lidar[0]
    visualizar_nube_puntos(puntos, remisiones, vis)

    # Iniciar el loop de visualización
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visor de archivos LiDAR 3D.')
    parser.add_argument('carpeta_lidar', type=str, help='Ruta a la carpeta que contiene los archivos LiDAR.')
    parser.add_argument('--tipo_archivo', choices=['bin', 'ply'], default='bin', help='Tipo de archivo a leer: bin o ply.')

    args = parser.parse_args()

    asyncio.run(main(args.carpeta_lidar, args.tipo_archivo))
