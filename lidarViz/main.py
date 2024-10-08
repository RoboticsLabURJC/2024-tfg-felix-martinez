import asyncio
from lector_lidar import leer_archivos_lidar
from visor_lidar import iniciar_visor

async def main():
    # Carpeta que contiene los archivos LiDAR
    carpeta_lidar = '/home/felix/Escritorio/TFG/datasets/Goose/goose_3d_val/lidar/val/2023-05-17_neubiberg_sunny/'
    
    # Leer todos los archivos LiDAR de forma asíncrona
    print("Leyendo archivos LiDAR...")
    datos_lidar = await leer_archivos_lidar(carpeta_lidar)  # Guardamos los datos en la variable
    
    # Iniciar el visor en el navegador pasando los datos procesados
    iniciar_visor(datos_lidar)  # Pasa la variable datos_lidar aquí

if __name__ == "__main__":
    asyncio.run(main())
