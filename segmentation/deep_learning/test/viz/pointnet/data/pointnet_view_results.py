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

def create_matplotlib_figure(real_path, pred_path, output_filename="comparacion_matplotlib.pdf", sample_ratio=1):
    """
    Carga, optimiza y visualiza dos nubes de puntos .ply lado a lado 
    en una figura de Matplotlib de alta calidad.
    """
    # --- Validación de Archivos ---
    for path in [real_path, pred_path]:
        if not os.path.exists(path):
            print(f"Error: No se encuentra el archivo: '{path}'")
            sys.exit(1)

    print(f"Cargando nube de puntos real desde: {real_path}")
    pcd_real = o3d.io.read_point_cloud(real_path)

    print(f"Cargando nube de puntos predicha desde: {pred_path}")
    pcd_pred = o3d.io.read_point_cloud(pred_path)

    if pcd_real.is_empty() or pcd_pred.is_empty():
        print("Error: Uno o ambos archivos .ply están vacíos.")
        sys.exit(1)

    # --- OPTIMIZACIÓN: Muestreo Aleatorio Uniforme ---
    # Matplotlib es lento con muchos puntos. Tomamos una muestra para agilizarlo.
    # sample_ratio: porcentaje de puntos a mantener (ej: 0.2 = 20%)
    num_points_to_sample_real = int(len(pcd_real.points) * sample_ratio)
    num_points_to_sample_pred = int(len(pcd_pred.points) * sample_ratio)

    print(f"\nOptimizando nubes de puntos con un muestreo del {sample_ratio * 100:.0f}%")
    print(f"Puntos originales (Real): {len(pcd_real.points)} -> Muestreados: {num_points_to_sample_real}")
    pcd_real_down = pcd_real.random_down_sample(sampling_ratio=sample_ratio)
    
    print(f"Puntos originales (Predicha): {len(pcd_pred.points)} -> Muestreados: {num_points_to_sample_pred}")
    pcd_pred_down = pcd_pred.random_down_sample(sampling_ratio=sample_ratio)
    
    # Extraer puntos y colores de las nubes optimizadas
    points_real = np.asarray(pcd_real_down.points)
    colors_real = np.asarray(pcd_real_down.colors)

    points_pred = np.asarray(pcd_pred_down.points)
    colors_pred = np.asarray(pcd_pred_down.colors)

    # --- Creación de la Figura de Matplotlib ---
    print("\nCreando figura con Matplotlib...")
    fig = plt.figure(figsize=(18, 8))

    # --- Subplot 1: Ground Truth ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(points_real[:, 0], points_real[:, 1], points_real[:, 2], c=colors_real, s=0.5)
    ax1.set_title('Ground Truth', fontsize=16)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.view_init(elev=20, azim=-60) # Vista inclinada
    ax1.grid(False) # Quitar el grid para un look más limpio
    # Forzar que los ejes tengan la misma escala
    axis_limits = np.array([points_real.min(axis=0), points_real.max(axis=0)]).T
    ax1.auto_scale_xyz(axis_limits[0], axis_limits[1], axis_limits[2])


    # --- Subplot 2: Prediction ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(points_pred[:, 0], points_pred[:, 1], points_pred[:, 2], c=colors_pred, s=0.5)
    ax2.set_title('Prediction', fontsize=16)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.view_init(elev=20, azim=-60)
    ax2.grid(False)
    axis_limits = np.array([points_pred.min(axis=0), points_pred.max(axis=0)]).T
    ax2.auto_scale_xyz(axis_limits[0], axis_limits[1], axis_limits[2])

    # --- Creación de la Leyenda ---
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=CLASS_NAMES[i],
                              markerfacecolor=COLOR_MAP_MPL[i], markersize=10) for i in CLASS_NAMES]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(CLASS_NAMES), bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Ajustar para que no se solape con la leyenda

    # --- Guardar y Mostrar la Figura ---
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nFigura guardada como: '{output_filename}'")
    
    plt.show()
    
if __name__ == "__main__":
    
    path_gt = r"C:\Users\felix\Desktop\2024-tfg-felix-martinez\segmentation\deep_learning\test\viz\pointnet\data\real_105.ply"
    path_pred = r"C:\Users\felix\Desktop\2024-tfg-felix-martinez\segmentation\deep_learning\test\viz\pointnet\data\predicha_105.ply"
    
    create_matplotlib_figure(path_gt, path_pred)
