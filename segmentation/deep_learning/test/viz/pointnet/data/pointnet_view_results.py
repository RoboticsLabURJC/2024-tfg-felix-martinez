import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
import os
import sys
import numpy as np
import open3d as o3d

# Definición de nombres y colores para la leyenda
CLASS_NAMES = {
    0: "Construction",
    1: "Object",
    2: "Road",
    3: "Sign",
    4: "Terrain",
    5: "Drivable Veg.",
    6: "Non-Drivable Veg.",
    7: "Vehicle",
    8: "Void"
}
COLOR_MAP_MPL = {
    0: "grey",
    1: "orange",
    2: "blue",
    3: "red",
    4: "brown",
    5: "lightgreen",
    6: "darkgreen",
    7: "yellow",
    8: "black"
}

def create_matplotlib_figure(paths, titles, output_filename="comparacion_modelos.pdf", sample_ratio=0.2):
    """
    Carga, optimiza y visualiza múltiples nubes de puntos .ply lado a lado 
    en una figura de Matplotlib de alta calidad.
    """
    point_clouds = []
    # --- Carga y Validación de Archivos ---
    for path in paths:
        if not os.path.exists(path):
            print(f"Error: No se encuentra el archivo: '{path}'")
            sys.exit(1)
        print(f"Cargando nube de puntos desde: {path}")
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            print(f"Error: El archivo .ply está vacío: {path}")
            sys.exit(1)
        point_clouds.append(pcd)

    # --- OPTIMIZACIÓN: Muestreo Aleatorio Uniforme ---
    print(f"\nOptimizando nubes de puntos con un muestreo del {sample_ratio * 100:.0f}%")
    downsampled_pcds = []
    for i, pcd in enumerate(point_clouds):
        num_points_to_sample = int(len(pcd.points) * sample_ratio)
        print(f"Nube '{titles[i]}': {len(pcd.points)} puntos -> Muestreados: {num_points_to_sample}")
        downsampled_pcds.append(pcd.random_down_sample(sampling_ratio=sample_ratio))
    
    # --- Creación de la Figura de Matplotlib ---
    print("\nCreando figura con Matplotlib...")
    fig = plt.figure(figsize=(24, 7)) # Figura más ancha para 4 subplots
    num_plots = len(paths)

    for i, (pcd_down, title) in enumerate(zip(downsampled_pcds, titles)):
        points = np.asarray(pcd_down.points)
        colors = np.asarray(pcd_down.colors)

        ax = fig.add_subplot(1, num_plots, i + 1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.5)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.view_init(elev=60, azim=-60)
        ax.grid(False)
        
        # Forzar que los ejes tengan la misma escala
        axis_limits = np.array([points.min(axis=0), points.max(axis=0)]).T
        ax.auto_scale_xyz(axis_limits[0], axis_limits[1], axis_limits[2])

    # --- Creación de la Leyenda ---
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=CLASS_NAMES[j],
                              markerfacecolor=COLOR_MAP_MPL[j], markersize=10) for j in CLASS_NAMES]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(CLASS_NAMES), bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0.05, 1, 0.9]) # Ajustar para que no se solapen los elementos

    # --- Guardar y Mostrar la Figura ---
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nFigura guardada como: '{output_filename}'")
    
    plt.show()
    
if __name__ == "__main__":
    
    path_gt = r"C:\Users\felix\Desktop\2024-tfg-felix-martinez\segmentation\deep_learning\test\viz\pointnet\data\real_300.ply"
    path_pred = r"C:\Users\felix\Desktop\2024-tfg-felix-martinez\segmentation\deep_learning\test\viz\pointnet\data\predicha_300.ply"
    path_pred = r"C:\Users\felix\Desktop\2024-tfg-felix-martinez\segmentation\deep_learning\test\viz\pointnet\data\predicha_300.ply"
    
    # Agrupar rutas y títulos para la función
    file_paths = [path_gt, path_pred, args.pointnetpp_file, args.pointnetpp_star_file]
    titles = ["Ground Truth", "PointNet", "PointNet++", "PointNet++*"]
    
    create_matplotlib_figure(path_gt, path_pred)
