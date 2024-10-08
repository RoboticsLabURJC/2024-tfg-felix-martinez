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

# Función asincrónica para leer el archivo de forma asíncrona usando asyncio.to_thread
async def leer_archivo_bin_async(filename):
    """
    Lee un archivo .bin de forma asincrónica utilizando asyncio.to_thread.
    
    :param filename: Ruta al archivo .bin.
    :return: Una tupla (points, remissions) con los datos leídos del archivo.
    """
    return await asyncio.to_thread(leer_archivo_bin, filename)

# Función para leer todos los archivos de una carpeta de forma asíncrona y ordenarlos por el número de frame
async def leer_archivos_lidar(carpeta):
    """
    Lee todos los archivos en la carpeta de forma asíncrona y los ordena por el número de frame.
    
    :param carpeta: Ruta a la carpeta con los archivos LiDAR.
    :return: Lista de tuplas con los datos de los archivos leídos (points, remissions) y sus nombres.
    """
    archivos = os.listdir(carpeta)  # Obtener los archivos en el orden del sistema de archivos

    # Expresión regular para extraer el framenumber (número de cuadro) del nombre del archivo
    def extraer_frame(archivo):
        match = re.search(r'__([0-9]{4})_', archivo)  # Busca el número de cuadro (4 dígitos) después de "__"
        if match:
            return int(match.group(1))  # Devuelve el número de frame como entero
        return -1  # Si no encuentra, devuelve un valor negativo para descartar archivos mal formateados

    # Ordenar los archivos según el número de frame extraído
    archivos = sorted(archivos, key=extraer_frame)

    # Crear las tareas de lectura asíncrona para los archivos ordenados
    tareas = []
    for archivo in archivos:
        ruta_completa = os.path.join(carpeta, archivo)
        print(f"Comenzando la lectura de {ruta_completa}")
        # Agregamos la tarea asíncrona para leer cada archivo
        tareas.append(leer_archivo_bin_async(ruta_completa))
    
    # Ejecutar todas las tareas de forma concurrente
    resultados = await asyncio.gather(*tareas)
    
    # Emparejar los resultados con los nombres de los archivos
    archivos_lidar = [(resultado, archivo) for resultado, archivo in zip(resultados, archivos)]
    
    return archivos_lidar  # Retorna una lista de tuplas (points, remissions, nombre de archivo)
