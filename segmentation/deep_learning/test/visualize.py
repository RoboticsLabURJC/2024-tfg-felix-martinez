import open3d as o3d
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Definición del Modelo y Dataset (Copiado de tu notebook) ---

class PointNet(nn.Module):
    """
    Definición de la arquitectura del modelo PointNet para segmentación semántica.
    """
    def __init__(self, num_classes=9):
        super(PointNet, self).__init__()
        # Capas convolucionales 1D para extracción de características
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Capa para predecir las clases
        self.out_conv = nn.Conv1d(1088, num_classes, 1)
        
        # Batch Normalization para estabilizar el entrenamiento
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Función de activación ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        # x tiene la forma [batch_size, 3, num_points]
        
        # Extracción de características locales
        x = self.relu(self.bn1(self.conv1(x)))
        point_features = x # Guardamos características para concatenar más tarde
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Max-pooling para obtener características globales
        global_feature = torch.max(x, 2, keepdim=True)[0]
        
        # Replicar la característica global para cada punto
        global_feature_repeated = global_feature.repeat(1, 1, x.size(2))
        
        # Concatenar características locales y globales
        combined_features = torch.cat([point_features, global_feature_repeated], 1)
        
        # Predicción final de clases
        x = self.out_conv(combined_features)
        
        # Reordenar la salida a [batch_size, num_points, num_classes]
        x = x.transpose(2, 1).contiguous()
        
        return x

class CustomDataset(Dataset):
    """
    Clase para cargar el dataset de SemanticKITTI.
    """
    def __init__(self, root_dir, sequences, num_points=16384):
        self.root_dir = root_dir
        self.sequences = sequences
        self.num_points = num_points
        self.file_paths = self._get_file_paths()

    def _get_file_paths(self):
        paths = []
        for seq in self.sequences:
            point_files = sorted(os.listdir(os.path.join(self.root_dir, seq, 'velodyne')))
            for fname in point_files:
                paths.append((os.path.join(self.root_dir, seq, 'velodyne', fname),
                              os.path.join(self.root_dir, seq, 'labels', fname.replace('.bin', '.label'))))
        return paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        point_path, label_path = self.file_paths[idx]
        points = np.fromfile(point_path, dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(label_path, dtype=np.int32)
        
        # Eliminar la reflectancia y la parte superior de la etiqueta
        points = points[:, :3]
        labels = labels & 0xFFFF 
        
        # Mapeo de etiquetas a las 9 clases
        # Este mapeo debe coincidir con el utilizado durante el entrenamiento
        label_mapping = {
            1: 0, 10: 0, 11: 0, 13: 0, 15: 0, 16: 0, 18: 0, 20: 0,
            30: 1, 31: 1, 32: 1,
            40: 2, 44: 2, 48: 2, 49: 2,
            50: 3, 51: 3,
            52: 4,
            60: 5,
            70: 6, 71: 6, 72: 6,
            80: 7, 81: 7,
            99: 8, 252: 0, 253: 6, 254: 5, 255: 8, 256: 4, 257: 4, 258: 6, 259: 0
        }
        mapped_labels = np.vectorize(label_mapping.get)(labels, 8) # '8' es la clase por defecto (vacío)
        
        # Submuestreo o padding de puntos
        if len(points) > self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(points), self.num_points, replace=True)
        
        points = points[choice, :]
        mapped_labels = mapped_labels[choice]
        
        return torch.from_numpy(points).float(), torch.from_numpy(mapped_labels).long()


# --- Configuración de Visualización ---
COLOR_MAP = {
    0: [0.5, 0.5, 0.5],       # Gris para Construcción
    1: [1.0, 0.5, 0.0],       # Naranja para Objeto
    2: [0.0, 0.0, 1.0],       # Azul para Carretera
    3: [1.0, 0.0, 0.0],       # Rojo para Señales
    4: [0.6, 0.4, 0.2],       # Marrón para Terreno
    5: [0.0, 1.0, 0.0],       # Verde claro para Vegetación transitable
    6: [0.0, 0.5, 0.0],       # Verde oscuro para Vegetación no transitable
    7: [1.0, 1.0, 0.0],       # Amarillo para Vehículo
    8: [0.0, 0.0, 0.0],       # Negro para Vacío
}

def get_color_for_label(label):
    return COLOR_MAP.get(label, [1.0, 1.0, 1.0])

def create_point_cloud(points, labels):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.array([get_color_for_label(l) for l in labels])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def visualize_point_clouds(points, real_labels, pred_labels, index):
    print(f"\nVisualizando el índice: {index}")
    real_pcd = create_point_cloud(points, real_labels)
    pred_pcd = create_point_cloud(points, pred_labels)
    
    o3d.visualization.draw_geometries([real_pcd], window_name=f"Nube Real - Índice {index}", width=800, height=600, left=50, top=50)
    o3d.visualization.draw_geometries([pred_pcd], window_name=f"Nube Predicha - Índice {index}", width=800, height=600, left=850, top=50)

# --- Función de Inferencia ---

def run_inference(model_path, data_root, test_sequences, num_points):
    """
    Ejecuta la inferencia en el conjunto de prueba y devuelve los resultados.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar dataset
    test_dataset = CustomDataset(root_dir=data_root, sequences=test_sequences, num_points=num_points)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Cargar modelo
    model = PointNet(num_classes=9).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del modelo en '{model_path}'")
        print("Por favor, asegúrate de que la ruta es correcta.")
        return None, None, None
        
    model.eval()

    all_points, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for points, labels in tqdm(test_loader, desc="Realizando inferencia"):
            points = points.to(device)
            
            # Transponer para que el modelo reciba [batch, features, points]
            points_transposed = points.transpose(1, 2)
            
            # Obtener predicciones del modelo
            outputs = model(points_transposed)
            preds = torch.argmax(outputs, dim=2)

            # Mover resultados a la CPU y convertir a NumPy
            all_points.append(points.cpu().numpy())
            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())

    # Concatenar resultados de todos los lotes
    all_points = np.concatenate(all_points, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    return all_points, all_labels, all_preds

def main():
    """
    Función principal para ejecutar la inferencia y la visualización.
    """
    # --- Configuración de Rutas ---
    # ¡¡¡IMPORTANTE!!! Modifica estas rutas para que apunten a tus archivos.
    MODEL_PATH = '../models/pointnet_semantickitti_9clases_v2.pth'
    DATA_ROOT_DIR = '../dataset_semantickitti/sequences'
    TEST_SEQUENCES = ['08'] # Usando la secuencia 08 como en tu notebook
    NUM_POINTS = 16384
    
    # 1. Ejecutar inferencia
    all_points, all_labels, all_preds = run_inference(
        model_path=MODEL_PATH,
        data_root=DATA_ROOT_DIR,
        test_sequences=TEST_SEQUENCES,
        num_points=NUM_POINTS
    )

    if all_points is None:
        return # Salir si la inferencia falló (p.ej., modelo no encontrado)
        
    num_muestras = all_points.shape[0]
    print(f"\nSe ha completado la inferencia. {num_muestras} muestras procesadas.")
    print("Iniciando modo de visualización interactiva.")

    # 2. Bucle de Visualización Interactivo
    while True:
        try:
            user_input = input(f"\nIntroduce un índice entre 0 y {num_muestras - 1} para visualizar (o escribe 'exit' para salir): ")
            
            if user_input.lower() == 'exit':
                break

            index_to_show = int(user_input)

            if 0 <= index_to_show < num_muestras:
                visualize_point_clouds(
                    all_points[index_to_show],
                    all_labels[index_to_show],
                    all_preds[index_to_show],
                    index_to_show
                )
            else:
                print(f"Índice fuera de rango. Por favor, introduce un número entre 0 y {num_muestras - 1}.")
        except ValueError:
            print("Entrada no válida. Por favor, introduce un número entero.")
        except Exception as e:
            print(f"Ha ocurrido un error durante la visualización: {e}")

if __name__ == "__main__":
    main()
