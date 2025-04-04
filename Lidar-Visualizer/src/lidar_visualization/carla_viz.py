
try:
    import carla
except:
    pass

import numpy as np
import open3d as o3d
import time
import random
from datetime import datetime
from matplotlib import colormaps as cm
import sys
import math
import pygame

VIRIDIS = np.array(cm.get_cmap('inferno').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

# Variables globales
actor_list = []
manual_mode = False
absolute_view = False
camera_views = [
    {"zoom": 0.3, "front": [0, 0, 1], "lookat": [0, 0, 0], "up": [-1, 0, 0]},  # Primera persona
    {"zoom": 0.06, "front": [1.0, 0.0, 0.3], "lookat": [0, 0, 0], "up": [0, 0, 1]},  # Tercera persona
    {"zoom": 0.2, "front": [0, 0, -1], "lookat": [0, 0, 0], "up": [0, 1, 0]}  # Vista cenital
]
current_view_index = 0  # Indice de la vista actual

def euler_to_rotation_matrix(pitch, yaw, roll):
    # Convertir los ángulos de grados a radianes
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    roll = math.radians(roll)

    # Crear la matriz de rotación para cada eje
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Multiplicar en el orden Yaw -> Pitch -> Roll
    rotation_matrix = R_z @ R_y @ R_x
    return rotation_matrix


initial_vehicle_location = None

def lidar_callback(lidar_data, point_cloud, vehicle_transform):
    global initial_vehicle_location

    # Leer los datos del LiDAR y reorganizarlos en forma de puntos e intensidad
    data = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # Reflejar los datos en el eje X para alinearlos con el sistema de CARLA
    data[:, 0] = -data[:, 0]

    if absolute_view:
        # Obtener ubicación y rotación del vehículo
        vehicle_location = np.array([-vehicle_transform.location.x,
                                     vehicle_transform.location.y,
                                     vehicle_transform.location.z])
        
        pitch = vehicle_transform.rotation.pitch
        yaw = vehicle_transform.rotation.yaw
        roll = vehicle_transform.rotation.roll

        # Crear la matriz de rotación inversa usando la orientación del vehículo
        rotation_matrix = euler_to_rotation_matrix(pitch, yaw, roll).T  # Usamos la transpuesta como la inversa

        # Aplicar la rotación inversa y luego posicionar en el sistema global
        points_in_world = []
        for point in data[:, :3]:  # 'data' contiene los puntos LiDAR relativos
            # Aplicar la rotación inversa al punto
            rotated_point = rotation_matrix @ point
            # Posicionar el punto en el sistema global usando la posición inicial del vehículo
            global_point = rotated_point + vehicle_location
            points_in_world.append(global_point)

        # Convertir a formato numpy para visualización en Open3D
        points_to_display = np.array(points_in_world)

    else:
        # Usar coordenadas relativas sin transformación
        points_to_display = data[:, :3]

    # Mapear la intensidad de cada punto a un color para visualización
    intensity = data[:, -1]
    int_color = np.c_[
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 2])
    ]

    # Actualizar el point cloud en Open3D
    point_cloud.points = o3d.utility.Vector3dVector(points_to_display)
    point_cloud.colors = o3d.utility.Vector3dVector(int_color)

def spawn_vehicle_lidar_camera(world, bp, traffic_manager, delta, lidar_range=100, channels=64, points_per_second=1200000):
    vehicle_bp = bp.filter('vehicle.*')[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    lidar_bp = bp.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', str(lidar_range))
    lidar_bp.set_attribute('rotation_frequency', str(1 / delta))
    lidar_bp.set_attribute('channels', str(channels))
    lidar_bp.set_attribute('points_per_second', str(points_per_second))
    lidar_position = carla.Transform(carla.Location(x=-0.5, z=1.8))
    lidar = world.spawn_actor(lidar_bp, lidar_position, attach_to=vehicle)

    camera_bp = bp.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=-4.0, z=2.5))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    vehicle.set_autopilot(True, traffic_manager.get_port())
    return vehicle, lidar, camera

def create_origin_sphere():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3)
    sphere.translate([0, 0, 0])  # Posicionar en el origen
    sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Color rojo
    return sphere

def set_camera_view(viz, view_index):
    ctr = viz.get_view_control()
    view = camera_views[view_index]
    ctr.set_zoom(view["zoom"])
    ctr.set_front(view["front"])
    ctr.set_lookat(view["lookat"])
    ctr.set_up(view["up"])

def camera_callback(image, display_surface):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Quitar el canal alfa
    array = array[:, :, ::-1]  # Convertir de BGRA a RGB
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display_surface.blit(surface, (0, 0))
    # Actualizar solo esta superficie en vez de toda la pantalla
    pygame.display.update(display_surface.get_rect())

def vehicle_control(vehicle):
    global manual_mode
    control = carla.VehicleControl()  # Crear un control en blanco

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            cleanup()
            sys.exit()

        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            manual_mode = not manual_mode
            vehicle.set_autopilot(not manual_mode)
            mode = "manual" if manual_mode else "automático"
            print(f"\nCoche en modo {mode}.")
            time.sleep(0.3)  # Evitar múltiples activaciones por pulsación rápida

    # Aplicar controles si estamos en modo manual
    keys = pygame.key.get_pressed()
    if manual_mode:
        control.throttle = 1.0 if keys[pygame.K_w] else 0.0
        control.brake = 1.0 if keys[pygame.K_s] else 0.0
        control.steer = -0.3 if keys[pygame.K_a] else 0.3 if keys[pygame.K_d] else 0.0
        vehicle.apply_control(control)

def main(lidar_range, channels, points_per_second):
    pygame.init()
    # Configurar Pygame sin usar OpenGL explícitamente
    pygame.display.gl_set_attribute(pygame.GL_ACCELERATED_VISUAL, 0)
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height), pygame.SRCALPHA)
    pygame.display.set_caption("CARLA Vehículo Control")

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    settings = world.get_settings()
    delta = 0.05
    settings.fixed_delta_seconds = delta
    settings.synchronous_mode = True
    world.apply_settings(settings)

    global actor_list, current_view_index, origin_sphere_added, absolute_view
    vehicle, lidar, camera = spawn_vehicle_lidar_camera(world, blueprint_library, traffic_manager, delta, lidar_range, channels, points_per_second)
    actor_list.append(vehicle)
    actor_list.append(lidar)
    actor_list.append(camera)

    camera.listen(lambda image: camera_callback(image, screen))

    point_cloud = o3d.geometry.PointCloud()
    lidar.listen(lambda data: lidar_callback(data, point_cloud, lidar.get_transform()))

    # Visualizador de Open3D y configuraciones iniciales
    viz = o3d.visualization.VisualizerWithKeyCallback()
    viz.create_window(window_name='Lidar simulado en Carla', width=960, height=540, left=480, top=270)
    viz.get_render_option().background_color = [0.05, 0.05, 0.05]
    viz.get_render_option().point_size = 1.35
    viz.get_render_option().show_coordinate_frame = True

    # Configuración de la vista inicial
    set_camera_view(viz, current_view_index)

    # Inicializa la esfera en el origen
    origin_sphere = create_origin_sphere()
    origin_sphere_added = False

    def toggle_camera_view(_):
        global current_view_index, absolute_view, origin_sphere_added
        # Alternar a la siguiente vista en el array de cámaras
        current_view_index = (current_view_index + 1) % len(camera_views)
        set_camera_view(viz, current_view_index)
        
        # Activar o desactivar vista absoluta y la esfera en el origen
        absolute_view = (current_view_index == len(camera_views) - 1)  # Vista cenital como absoluta
        if absolute_view and not origin_sphere_added:
            viz.add_geometry(origin_sphere)
            origin_sphere_added = True
        elif not absolute_view and origin_sphere_added:
            viz.remove_geometry(origin_sphere)
            origin_sphere_added = False
        
        print(f"Cambiando a vista {current_view_index + 1}")
        return True

    # Callback para alternar vistas
    viz.register_key_callback(ord("V"), toggle_camera_view)

    # Ciclo principal de visualización
    lidar_data_received = False
    dt0 = datetime.now()
    frame = 0

    while True:
        vehicle_control(vehicle)

        if frame == 5 and not lidar_data_received:
            viz.add_geometry(point_cloud)
            lidar_data_received = True
            print("Geometry added to the visualizer")

        viz.update_geometry(point_cloud)
        viz.poll_events()
        viz.update_renderer()
        time.sleep(0.03)
        world.tick()

        process_time = datetime.now() - dt0
        if process_time.total_seconds() > 0:
            fps = 1.0 / process_time.total_seconds()
            sys.stdout.write(f'\rFPS: {fps:.2f} ')
            sys.stdout.flush()
        dt0 = datetime.now()
        frame += 1

        if not viz.poll_events():
            print("Exiting visualization")
            break

    cleanup()

def cleanup():
    global actor_list
    print("\nLimpiando actores...")
    for actor in actor_list:
        if actor is not None:
            actor.destroy()
    actor_list = []
    print("Actores eliminados.")

def signal_handler(sig, frame):
    print("\nInterrupción recibida. Finalizando...")
    cleanup()
    sys.exit(0)