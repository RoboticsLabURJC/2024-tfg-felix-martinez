import os
import re
import numpy as np
import asyncio

# Función síncrona para leer un archivo .bin
def leer_archivo_bin(filename):
    """
    Lee un archivo .bin y devuelve los puntos y remisiones.
    """
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, 0:3]    # Coordenadas XYZ
    remissions = scan[:, 3]  # Intensidades de remisión
    return points[::8], remissions[::8]  # Muestreo para optimizar rendimiento

# Función asincrónica para leer los archivos LiDAR de manera ordenada
async def leer_archivos_lidar(carpeta):
    """
    Lee todos los archivos en la carpeta de forma asíncrona y los ordena por el número de frame.
    
    :param carpeta: Ruta a la carpeta con los archivos LiDAR.
    :return: Lista de tuplas con los datos de los archivos leídos (points, remissions) y sus nombres.
    """
    archivos = os.listdir(carpeta)

    # Ordenar los archivos LiDAR por número de frame
    def extraer_frame(archivo):
        match = re.search(r'__([0-9]{4})_', archivo)  # Busca el número de cuadro (4 dígitos) después de "__"
        if match:
            return int(match.group(1))  # Devuelve el número de frame como entero
        return -1  # Si no encuentra, devuelve un valor negativo para descartar archivos mal formateados

    archivos = sorted(archivos, key=extraer_frame)

    # Leer los archivos LiDAR
    tareas = []
    for archivo in archivos:
        ruta_completa = os.path.join(carpeta, archivo)
        print(f"Leyendo archivo: {ruta_completa}")
        tareas.append(leer_archivo_bin(ruta_completa))
    
    return tareas  # Retorna la lista de tuplas (puntos, remisiones)
    
