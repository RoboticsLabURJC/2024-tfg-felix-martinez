import os
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
    return points[::5], remissions[::5] # Muestreo 1/5 para aunmentar rendimiento

# Función asincrónica para leer el archivo de forma asíncrona usando asyncio.to_thread
async def leer_archivo_bin_async(filename):
    """
    Lee un archivo .bin de forma asincrónica utilizando asyncio.to_thread.
    
    :param filename: Ruta al archivo .bin.
    :return: Una tupla (points, remissions) con los datos leídos del archivo.
    """
    return await asyncio.to_thread(leer_archivo_bin, filename)

# Función para leer todos los archivos de una carpeta de forma asíncrona, sin ordenar
async def leer_archivos_lidar(carpeta):
    """
    Lee todos los archivos en la carpeta de forma asíncrona tal como están en el sistema de archivos.
    
    :param carpeta: Ruta a la carpeta con los archivos LiDAR.
    :return: Lista de tuplas con los datos de los archivos leídos (points, remissions).
    """
    archivos = os.listdir(carpeta)  # Obtener los archivos en el orden del sistema de archivos
    
    tareas = []
    for archivo in archivos:
        ruta_completa = os.path.join(carpeta, archivo)
        print(f"Comenzando la lectura de {ruta_completa}")
        # Agregamos la tarea asíncrona para leer cada archivo
        tareas.append(leer_archivo_bin_async(ruta_completa))
    
    # Ejecutar todas las tareas de forma concurrente
    resultados = await asyncio.gather(*tareas)
    
    return resultados  # Retorna una lista con los puntos y remisiones de cada archivo
