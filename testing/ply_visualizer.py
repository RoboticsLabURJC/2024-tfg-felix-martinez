import open3d as o3d
import numpy as np
import os
import asyncio
import matplotlib.pyplot as plt
from plyfile import PlyData
import matplotlib.cm as cm

# Colormaps disponibles
colormaps = ['jet', 'viridis', 'plasma', 'inferno']  # Diferentes mapas de colores
colormap_idx = [0]  # Índice mutable para cambiar el mapa

# Estado para vista reducida o completa
mostrar_reducido = [False]

# Función para leer un archivo .ply
def leer_archivo_ply(ruta_archivo):
    """
    Lee un archivo .ply y devuelve las coordenadas XYZ y la intensidad.
    """
    print(f"[DEBUG] Leyendo archivo PLY: {ruta_archivo}")
    
    # Leer el archivo PLY usando plyfile
    plydata = PlyData.read(ruta_archivo)

    # Extraer las coordenadas x, y, z
    x = plydata['vertex'].data['x']
    y = plydata['vertex'].data['y']
    z = plydata['vertex'].data['z']
    puntos = np.vstack((x, y, z)).T  # Matriz de puntos [N, 3] con x, y, z

    # Extraer la intensidad
    intensidad = plydata['vertex'].data['intensity']

    return puntos, intensidad

# Función para visualizar una nube de puntos en Open3D
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

# Función para actualizar la nube de puntos sin restaurar la cámara
def actualizar_nube_de_puntos(vis, puntos, remisiones):
    # Actualizar la nube de puntos sin intentar restaurar el estado de la cámara
    visualizar_nube_puntos(puntos, remisiones, vis)

# Función principal
async def main(carpeta):
    # Obtener la lista de archivos .ply en la carpeta
    archivos = sorted([f for f in os.listdir(carpeta) if f.endswith('.ply')])
    
    if not archivos:
        print("[ADVERTENCIA] No se encontraron archivos .ply en la carpeta.")
        return
    
    print(f"[INFO] Se encontraron {len(archivos)} archivos .ply")

    # Crear una ventana de visualización Open3D
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()  # Sin fijar el tamaño de la ventana

    # Crear una geometría de nube de puntos vacía
    nube_puntos = o3d.geometry.PointCloud()
    vis.add_geometry(nube_puntos)

    # Obtener las opciones de renderizado para ajustar el tamaño de los puntos
    render_options = vis.get_render_option()
    render_options.point_size = 2.0  # Tamaño pequeño para los puntos
    render_options.light_on = False  # Desactivar la iluminación

    # Índice actual y máximo
    index_actual = [0]  # Usamos una lista mutable para poder modificarlo dentro de los callbacks
    max_index = len(archivos) - 1

    # Mostrar la primera nube de puntos
    ruta_completa = os.path.join(carpeta, archivos[index_actual[0]])
    puntos, remisiones = leer_archivo_ply(ruta_completa)
    visualizar_nube_puntos(puntos, remisiones, vis)

    # Callback para avanzar (tecla flecha derecha)
    def avanzar(vis):
        if index_actual[0] < max_index:
            index_actual[0] += 1
            ruta_completa = os.path.join(carpeta, archivos[index_actual[0]])
            puntos, remisiones = leer_archivo_ply(ruta_completa)
            actualizar_nube_de_puntos(vis, puntos, remisiones)
            print(f"Mostrando archivo {index_actual[0] + 1}/{max_index + 1}")
        return False

    # Callback para retroceder (tecla flecha izquierda)
    def retroceder(vis):
        if index_actual[0] > 0:
            index_actual[0] -= 1
            ruta_completa = os.path.join(carpeta, archivos[index_actual[0]])
            puntos, remisiones = leer_archivo_ply(ruta_completa)
            actualizar_nube_de_puntos(vis, puntos, remisiones)
            print(f"Mostrando archivo {index_actual[0] + 1}/{max_index + 1}")
        return False

    # Callback para cambiar el colormap con la tecla 'C'
    def cambiar_colormap(vis):
        colormap_idx[0] = (colormap_idx[0] + 1) % len(colormaps)
        print(f"Usando colormap: {colormaps[colormap_idx[0]]}")
        puntos, remisiones = leer_archivo_ply(os.path.join(carpeta, archivos[index_actual[0]]))
        actualizar_nube_de_puntos(vis, puntos, remisiones)
        return False

    # Callback para alternar entre vista completa y reducida con la tecla 'R'
    def alternar_vista_reducida(vis):
        mostrar_reducido[0] = not mostrar_reducido[0]
        puntos, remisiones = leer_archivo_ply(os.path.join(carpeta, archivos[index_actual[0]]))
        if mostrar_reducido[0]:
            idx = np.random.choice(len(puntos), size=len(puntos) // 10, replace=False)
            puntos_reducidos = puntos[idx]
            remisiones_reducidas = remisiones[idx]
            print("Mostrando vista reducida (10% de los puntos)")
            actualizar_nube_de_puntos(vis, puntos_reducidos, remisiones_reducidas)
        else:
            actualizar_nube_de_puntos(vis, puntos, remisiones)
            print("Mostrando vista completa")
        return False

    # Asignar los callbacks de las teclas
    vis.register_key_callback(262, avanzar)  # Tecla flecha derecha (código ASCII: 262)
    vis.register_key_callback(263, retroceder)  # Tecla flecha izquierda (código ASCII: 263)
    vis.register_key_callback(67, cambiar_colormap)  # Tecla 'C' para cambiar el colormap
    vis.register_key_callback(82, alternar_vista_reducida)  # Tecla 'R' para alternar vista reducida/completa

    # Iniciar el loop de visualización
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    carpeta = "/home/felix/Escritorio/TFG/datasets/Rellis_3D_os1_cloud_node_color_ply/Rellis-3D/00000/os1_cloud_node_color_ply"
    asyncio.run(main(carpeta))
