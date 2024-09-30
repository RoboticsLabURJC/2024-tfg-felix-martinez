import carla
import pygame

# --------------- Conectar al Servidor principal ------------------ #

# Conectarse al servidor principal de CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)  # Tiempo límite para conectarse

# Obtener el MUNDO del simulador obtenido de la conexión con el servidor
world = client.get_world()

# Obtener Blueprint (Lista de actores disponibles en el Mundo conectado)
blueprint_library = world.get_blueprint_library()

# ----------------  Agregar un ACTOR Vehículo  -------------------- #

# Filtramos por tipo de actores: Vehiculo y seleccionamos el primero de la lista resultante
vehicle_00 = blueprint_library.filter('vehicle.*')[0]

# Obtener los puntos de Spawn del MAPA y elegir el primero de la lista resultante
spawn_point = world.get_map().get_spawn_points()[0]

# Spawnear el objeto de Tipo: Actor_Vehiculo (Se crea un nuevo objeto al generarlo)
vehicle = world.spawn_actor(vehicle_00, spawn_point)

# Activar el modo de conducción automática por defecto que odrece carla con el metodo set_autopilot(True)
vehicle.set_autopilot(True)

# ---------------- Agregar un ACTOR Sensor LiDAR y configurar sus atributos -------------------- #

# Filtramos por tipo de actores: Sensores, lidar.raycast (Sensor LiDAR simulado en Carla)
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

# Configuración del objeto obtenido de tipo actor_sensor_lidar.raycast

# Configurar Rango MAX del sensor
lidar_bp.set_attribute('range', '40')
# Configurar la Frecuencia de rotación en Hz
lidar_bp.set_attribute('rotation_frequency', '10')
# Configurar número de canales
lidar_bp.set_attribute('channels', '32')
# Configurar frecuencia de muestreo (56 kHz)
lidar_bp.set_attribute('points_per_second', '56000')

# Ubicar el objeto en el espacio (teniendo en cuenta que luego se acoplará al vehículo, solo altura)
lidar_position = carla.Transform(carla.Location(x=0, z=2.5))

# Spawnear el sensor y acoplarlo al objeto de Tipo Actor_Vehiculo
# El tercer parámetro vincula la posición del actor LiDAR a otro objeto de tipo Actor_Vehiculo.
lidar = world.spawn_actor(lidar_bp, lidar_position, attach_to=vehicle)

# Función de callback para procesar los datos LiDAR
def lidar_callback(point_cloud):
    print("LiDAR data received: ", point_cloud)

# Activación del método listen() del actor_sensor_lidar.raycast
# "Todos los Actores de Tipo Sensor tienen el método listen() para obtener la información"
lidar.listen(lidar_callback)

# -------------------- Logica de lanzamiento ----------------- # 
try:
    # Mantener el script corriendo
    while True:
        world.wait_for_tick()
except KeyboardInterrupt:
    pass
finally:
    # Limpiar objetos después de cerrar el script

    # Parar de escuchar
    lidar.stop()
    # Destruir objeto spawn_sensor
    lidar.destroy()
    # Destruir objeto spawn_vehiculo
    vehicle.destroy()