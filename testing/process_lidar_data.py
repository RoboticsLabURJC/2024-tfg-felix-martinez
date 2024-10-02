import numpy as np

# Función para leer nube de puntos LiDAR desde un archivo .bin
def read_lidar_bin_file(bin_file):
    # Lee el archivo binario (.bin) como un array de números flotantes (float32)
    scan = np.fromfile(bin_file, dtype=np.float32)
    # Reorganiza los datos en filas de 4 columnas (x, y, z, intensidad)
    scan = scan.reshape((-1, 4))
    # Separa las columnas en las coordenadas xyz y la intensidad
    points = scan[:, 0:3]  # coordenadas XYZ
    intensities = scan[:, 3]  # valores de intensidad
    return points, intensities

# Función principal para procesar un archivo específico de LiDAR
def process_lidar_file(file_path):
    # Llamamos a la función para leer los puntos LiDAR
    points, intensities = read_lidar_bin_file(file_path)
    
    # Muestra información básica sobre los datos
    print(f"Total de puntos: {points.shape[0]}")
    print(f"Puntos: {points[5000:10000]}")
    print(f"Intensidades: {intensities}")
    
    # Aquí puedes realizar más operaciones con los puntos e intensidades

# Ruta al archivo .bin que deseas procesar (cámbiala según tu ubicación)
file_path = "/home/felix/Escritorio/TFG/datasets/Goose/goose_3d_val/lidar/val/2022-07-22_flight/2022-07-22_flight__0195_1658494596120446131_vls128.bin"


# Llamada para procesar el archivo
process_lidar_file(file_path)
