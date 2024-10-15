import carla
import numpy as np
import open3d as o3d
import time
import queue
import threading

# Cola para almacenar los datos LiDAR
lidar_queue = queue.Queue()

# Función para procesar los datos LiDAR y almacenarlos en la cola
def lidar_callback(lidar_data):
    points = []
    for point in lidar_data:
        points.append([point.point.x, point.point.y, point.point.z])

    if len(points) > 0:
        print(f"Recibidos {len(points)} puntos LiDAR.")  # Imprime el número de puntos recibidos
        points = np.array(points)
        lidar_queue.put(points)  # Guardar los puntos en la cola
    else:
        print("Advertencia: No se han recibido puntos del LiDAR.")

# Función para spawnear el vehículo y el sensor LiDAR
def spawn_vehicle_and_lidar(world):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_points = world.get_map().get_spawn_points()

    vehicle = None
    for spawn_point in spawn_points:
        try:
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            print(f"Vehículo creado en {spawn_point.location}")
            vehicle.set_autopilot(True)  # Activar el piloto automático
            break
        except RuntimeError:
            print(f"Fallo al crear vehículo en {spawn_point.location}, probando otro punto...")

    if not vehicle:
        raise RuntimeError("No se pudo spawnear el vehículo en ningún punto disponible")

     # Configurar el sensor LiDAR
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('lower_fov', '-20.0')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('rotation_frequency', str(1 / 0.05))  # 10 Hz
    lidar_bp.set_attribute('points_per_second', '400000')  # 100,000 puntos por segundo

    lidar_spawn_point = carla.Transform(carla.Location(x=0, z=1.8))  # Posición del LiDAR
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_spawn_point, attach_to=vehicle)
    print("Sensor LiDAR creado y adjunto al vehículo")

    return vehicle, lidar_sensor

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Crear el visualizador de Open3D
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Visualización de LiDAR", width=800, height=600)

    # Crear una nube de puntos vacía para actualizarla con datos LiDAR
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    render_options = vis.get_render_option()
    render_options.point_size = 2  # Tamaño pequeño para los puntos

    # Añadir ejes de referencia para visualizar el espacio
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
    vis.add_geometry(axes)

    # Configurar la cámara para vista de planta y zoom
    ctr = vis.get_view_control()
    ctr.set_zoom(15)  # Ajustar el zoom
    ctr.set_front([0, 0, 1])  # Vista desde arriba (planta)
    ctr.set_lookat([0, 0, 0])  # El punto de interés (centro de la escena)
    ctr.set_up([0, -1, 0])  # Definir el eje "arriba" para la cámara

    # Spawnear vehículo y añadir sensor LiDAR
    vehicle, lidar_sensor = spawn_vehicle_and_lidar(world)

    # Configurar el callback en un hilo separado para recibir datos LiDAR
    lidar_thread = threading.Thread(target=lambda: lidar_sensor.listen(lidar_callback))
    lidar_thread.start()

    try:
        print("Recolectando y visualizando datos LiDAR en vivo...")

        while True:
            # Procesar datos de la cola y actualizar la visualización
            if not lidar_queue.empty():
                points = lidar_queue.get()  # Obtener los datos LiDAR de la cola
                pcd.points = o3d.utility.Vector3dVector(points)  # Actualizar la nube de puntos

                # Forzar actualización de la geometría y visualización
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

            time.sleep(0.1)  # Pequeña pausa para evitar sobrecarga del bucle
    except KeyboardInterrupt:
        print("Interrumpido por el usuario. Deteniendo la simulación.")
    finally:
        lidar_sensor.stop()
        vehicle.destroy()
        vis.destroy_window()

if __name__ == "__main__":
    main()
