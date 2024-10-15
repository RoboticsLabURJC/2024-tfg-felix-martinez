import carla
import numpy as np
import open3d as o3d
import time
import random
from datetime import datetime
from matplotlib import colormaps as cm
import sys
import signal

VIRIDIS = np.array(cm.get_cmap('inferno').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

def lidar_callback(lidar_data, point_cloud):
    '''
    Procesa los datos brutos de carla 
    cada vez que se toma una muestra (callback) 
    y actualiza la nube de puntos en el objeto PointCloud 
    '''
    data = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    intensity = data[:, -1]
    int_color = np.c_[
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 2])]

    # Convertir la nube de puntos al formato de Open3D
    point_cloud.points = o3d.utility.Vector3dVector(data[:, :-1])
    point_cloud.colors = o3d.utility.Vector3dVector(int_color)

def spawn_vehicle_lidar(world, bp):
    vehicle_00 = bp.filter('vehicle.*')[0]
    spawn_points = world.get_map().get_spawn_points()

    spawn = False
    while True:
        if not spawn:
            spawn_point = random.choice(spawn_points)
            if not spawn_point:
                pass
            else:
                break

    vehicle = world.spawn_actor(vehicle_00, spawn_point)
    vehicle.set_autopilot(True)

    lidar_bp = bp.find('sensor.lidar.ray_cast')

    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('rotation_frequency', '60')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '500000')

    lidar_position = carla.Transform(carla.Location(x=-0.5, z=1.8))
    lidar = world.spawn_actor(lidar_bp, lidar_position, attach_to=vehicle)
    
    return vehicle, lidar

def add_open3d_axis(vis):
    """
    Añade un pequeño 3D axis en el Open3D Visualizer
    """
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)

# Variables globales para almacenar actores
actor_list = []

def cleanup():
    """
    Elimina todos los actores de Carla (vehículos, sensores, etc.)
    """
    global actor_list
    print("\nLimpiando actores...")
    for actor in actor_list:
        if actor is not None:
            actor.destroy()
    actor_list = []
    print("Actores eliminados.")

def signal_handler(sig, frame):
    """
    Captura la señal de interrupción (Ctrl+C) y limpia los actores antes de salir.
    """
    print("\nInterrupción recibida. Finalizando...")
    cleanup()
    sys.exit(0)

def main():
    '''
    Funcion Main del programa
    '''
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # Tiempo límite para conectarse

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    settings = world.get_settings()
    settings.synchronous_mode = True  # Activar modo sincronizado para modo sin renderizar
    world.apply_settings(settings)

    # Global para evitar que los actores se eliminen automáticamente
    global actor_list
    vehicle, lidar = spawn_vehicle_lidar(world, blueprint_library)
    actor_list.append(vehicle)
    actor_list.append(lidar)

    point_cloud = o3d.geometry.PointCloud()

    lidar.listen(lambda data: lidar_callback(data, point_cloud))

    viz = o3d.visualization.Visualizer()
    viz.create_window(
            window_name='Lidar simulado en Carla',
            width=960,
            height=540,
            left=480,
            top=270)
    viz.get_render_option().background_color = [0.05, 0.05, 0.05]
    viz.get_render_option().point_size = 1.35
    viz.get_render_option().show_coordinate_frame = True

    add_open3d_axis(viz)

    frame = 0
    dt0 = datetime.now()
    lidar_data_received = False  # Verificar si se recibe data de LiDAR

    while True:
        if frame == 5 and not lidar_data_received: # Pequeño buffer para que no colapse el visualizador
            # Añadir la nube de puntos solo después de recibir los datos
            viz.add_geometry(point_cloud)
            lidar_data_received = True  # Marca que hemos recibido datos
            print("Geometry added to the visualizer")

        # Actualizamos la geometría y nos aseguramos de que los puntos sigan siendo negros
        viz.update_geometry(point_cloud)

        viz.poll_events() # Sondear eventos de usuario para mantener la interactividad fluida en el bucle (movimientos de camara, etc)
        viz.update_renderer() # Actualizar el renderizado con datos nuevos

        time.sleep(0.005) # Tiempo de espera para sincronismo
        world.tick() # Avanzar un frame en el simulador

        # Calcular el tiempo de procesamiento para determinar los FPS
        process_time = datetime.now() - dt0
        if process_time.total_seconds() > 0:  # Evitar divisiones por cero
            fps = 1.0 / process_time.total_seconds()
            # Actualizar los FPS en la misma línea
            sys.stdout.write(f'\rFPS: {fps:.2f} ')
            sys.stdout.flush()
        dt0 = datetime.now()
        frame += 1

        # Condición de salida para terminar de forma segura
        if not viz.poll_events():
            print("Exiting visualization")
            break

    cleanup()  # Asegurarse de limpiar los actores al salir del ciclo principal
    
if __name__ == "__main__":
    # Capturar señales de interrupción (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    
    main()
