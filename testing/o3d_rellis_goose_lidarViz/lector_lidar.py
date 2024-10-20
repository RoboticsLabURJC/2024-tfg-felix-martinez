import os
import re
import numpy as np
import open3d as o3d
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

# Función síncrona para leer un archivo .ply
def leer_archivo_ply(filename):
    """
    Lee un archivo .ply y devuelve los puntos y remisiones.
    """
    nube_puntos = o3d.io.read_point_cloud(filename)
    puntos = np.asarray(nube_puntos.points)
    
    # Si el archivo .ply tiene un canal de intensidades de remisión
    if hasattr(nube_puntos, 'intensity'):
        remisiones = np.asarray(nube_puntos.intensity)
    else:
        remisiones = np.zeros(puntos.shape[0])  # Si no tiene, usa ceros

    return puntos, remisiones

# Función para extraer el número de frame de archivos Goose o Rellis
def extraer_frame(archivo):
    """
    Extrae el número de frame de un archivo.
    - Para Goose: busca "__XXXX_" (donde XXXX son 4 dígitos).
    - Para Rellis: busca "frameXXXXXX" (donde XXXXXX son dígitos).
    """
    # Condición para Goose (patrón: "__XXXX_")
    match_goose = re.search(r'__([0-9]{4})_', archivo)
    if match_goose:
        return int(match_goose.group(1))  # Devuelve el número de frame como entero
    
    # Condición para Rellis 3D (patrón: "frameXXXXXX")
    match_rellis = re.search(r'frame([0-9]+)', archivo)
    if match_rellis:
        return int(match_rellis.group(1))  # Devuelve el número de frame como entero

    return -1  # Si no encuentra, devuelve un valor negativo para descartar archivos mal formateados

# Función asincrónica para leer un archivo en paralelo con control de concurrencia
async def leer_archivo_lidar(ruta_completa, tipo_archivo, semaforo):
    """
    Función asíncrona para leer un archivo LiDAR de forma paralela con un semáforo para controlar la concurrencia.
    Muestra un mensaje de progreso cuando se completa la lectura de un archivo.
    """
    async with semaforo:
        if tipo_archivo == 'bin':
            datos = leer_archivo_bin(ruta_completa)
        elif tipo_archivo == 'ply':
            datos = leer_archivo_ply(ruta_completa)
        
    # Mostrar mensaje de progreso después de liberar el semáforo
    print(f"[INFO] Archivo procesado: {os.path.basename(ruta_completa)}")
        
    return datos

async def leer_archivos_lidar(carpeta, tipo_archivo='bin', limite_concurrencia=5):
    """
    Lee todos los archivos en la carpeta de forma asincrónica en paralelo, con un límite en la cantidad de archivos
    que se leen simultáneamente, y luego los ordena por el número de frame.
    """
    print(f"[DEBUG] Intentando listar archivos en la carpeta: {carpeta}")

    archivos = os.listdir(carpeta)
    
    # Mostrar el número de archivos encontrados
    print(f"[INFO] Archivos encontrados en la carpeta: {len(archivos)}")

    # Filtrar los archivos según el tipo
    if tipo_archivo == 'bin':
        archivos = [f for f in archivos if f.endswith('.bin')]
    elif tipo_archivo == 'ply':
        archivos = [f for f in archivos if f.endswith('.ply')]

    # Mostrar el número de archivos después del filtro
    print(f"[INFO] Archivos .{tipo_archivo} encontrados: {len(archivos)}")

    if not archivos:
        print("[ADVERTENCIA] No se encontraron archivos con la extensión especificada.")
        return []

    # Preparar las rutas completas
    rutas_completas = [os.path.join(carpeta, archivo) for archivo in archivos]

    # Crear un semáforo para limitar el número de lecturas simultáneas
    semaforo = asyncio.Semaphore(limite_concurrencia)

    # Mostrar mensaje de creación de tareas
    print(f"[DEBUG] Creando tareas para {len(rutas_completas)} archivos...")

    # Crear tareas para leer los archivos en paralelo con el semáforo
    tareas = [leer_archivo_lidar(ruta, tipo_archivo, semaforo) for ruta in rutas_completas]

    # Ejecutar las tareas de lectura en paralelo
    print(f"[DEBUG] Ejecutando tareas en paralelo...")
    resultados = await asyncio.gather(*tareas)

    # Emparejar los resultados con los nombres de archivo
    archivos_leidos = [(archivo, resultado) for archivo, resultado in zip(archivos, resultados)]

    # Ordenar los resultados por el número de frame
    archivos_ordenados = sorted(archivos_leidos, key=lambda x: extraer_frame(x[0]))

    # Devolver solo los puntos y remisiones ordenados
    datos_ordenados = [resultado for _, resultado in archivos_ordenados]

    print(f"[INFO] Lectura y procesamiento completado para {len(datos_ordenados)} archivos.")
    
    return datos_ordenados

