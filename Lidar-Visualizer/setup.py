from setuptools import setup, find_packages

setup(
    name="lidar_visualizer",
    version="0.2",
    packages=find_packages(where="src"),  # Busca los paquetes en la carpeta 'src'
    package_dir={"": "src"},              # Indica que el directorio base para paquetes es 'src'
    install_requires=[
        "open3d",
        "numpy",
        "matplotlib",
        "plyfile",
        "pygame",
        "carla",
        "tk",  # En algunos casos, `tk` puede no instalarse, ya que suele venir con Python
    ],
    entry_points={
        "console_scripts": [
            "lidar-viz=main:main",  # Crea un comando 'lidar-viz' para ejecutar el programa
        ],
    },
    include_package_data=True,  # Incluye datos adicionales, si los hay
    description="Proyecto de visualización de datos LiDAR con soporte para archivos y simulador CARLA",
    author="Félix Martínez",
    author_email="felixmaral131@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)