import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

MAX_POINTS = 14000

data_path = '/home/felix/Escritorio/TFG/datasets/Goose/goose_3d_val/lidar/val/2022-07-22_flight/'
labels_path = '/home/felix/Escritorio/TFG/datasets/Goose/goose_3d_val/labels/val/2022-07-22_flight/'

files_list = os.listdir(data_path)
labels_list = os.listdir(labels_path)

# Ordenar los archivos de puntos y etiquetas por el número en su nombre
sorted_files = sorted(files_list, key=lambda x: int(re.search(r'__\d{4,5}_', x).group(0)[2:-1]))
sorted_labels = sorted(labels_list, key=lambda x: int(re.search(r'__\d{4,5}_', x).group(0)[2:-1]))

points_dfs = [] # array de dataframes donde irá cada muestra Lidar

# Procesar archivos emparejados

for file, label_file in zip(sorted_files, sorted_labels):
    # Cargar el archivo de puntos
    scan = np.fromfile(data_path + file, dtype=np.float32).reshape((-1, 4))

    # Separar las columnas en xyz y remisiones
    points = scan[:, 0:3][0:MAX_POINTS]   # Coordenadas XYZ
    remissions = scan[:, 3][0:MAX_POINTS]  # Remisiones

    # Cargar el archivo de etiquetas correspondiente
    label = np.fromfile(labels_path + label_file, dtype=np.uint32).reshape((-1))[0:MAX_POINTS]

    # Validar tamaños
    if len(points) != len(label):
        print(f"Error en {file} y {label_file}: Points({len(points)}) != Labels({len(label)})")
        continue

    # Extraer etiquetas semánticas
    sem_label = label & 0xFFFF  # Extracción de los 4 priemros bytes (identificador de clase)  

    # Crear un nuevo DataFrame para este archivo
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df['remissions'] = remissions
    df['label'] = sem_label

    # Agregar el DataFrame a la lista
    points_dfs.append(df)
    print(f'\r{sorted_files.index(file)}/{len(sorted_files)}', end='', flush=True) # Progreso de la carga de datos

combined_df = pd.concat([df[['x', 'y', 'z', 'remissions']] for df in points_dfs]) # concatenar todos los df en uno para normalización

# Calcular los mínimos y máximos globales para las columnas numéricas
global_min = combined_df.min()
global_max = combined_df.max()

normalized_dfs = []

for df in points_dfs:
    # Copia el DataFrame para no modificar el original
    normalized_df = df.copy()

    # Normalizar solo las columnas numéricas ('x', 'y', 'z', 'remissions')
    normalized_df[['x', 'y', 'z', 'remissions']] = (
        df[['x', 'y', 'z', 'remissions']] - global_min[['x', 'y', 'z', 'remissions']]
    ) / (global_max[['x', 'y', 'z', 'remissions']] - global_min[['x', 'y', 'z', 'remissions']])

    # Mantener la columna 'label' sin cambios
    normalized_df['label'] = df['label']

    # Agregar el DataFrame normalizado a la lista
    normalized_dfs.append(normalized_df)
    
normalized_npa = np.asarray(normalized_dfs)

# Extraer las nubes de puntos (sin las etiquetas)
points = normalized_npa[:, :, :-1]  # Forma resultante: (151, 14000, 4)

# Extraer las etiquetas en un array separado
labels = normalized_npa[:, :, -1]  # Forma resultante: (151, 14000)

# Dividimos en entrenamiento y validación
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(points, labels, test_size=0.2, random_state=42)

# -----------------------------
# DEFINICIÓN DEL MODELO POINTNET
# -----------------------------
def build_pointnet(num_classes, input_dim=4):
    """
    Modelo PointNet para segmentación punto a punto.
    """
    inputs = tf.keras.Input(shape=(None, input_dim))  # Entrada: N puntos con D características

    # Capa convolucional inicial (clasificación)
    x = layers.Conv1D(64, 1, activation='relu')(inputs)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.Conv1D(1024, 1, activation='relu')(x)

    # Global Feature Aggregation
    global_features = layers.GlobalMaxPooling1D()(x)
    global_features = layers.RepeatVector(MAX_POINTS)(global_features)  # Repetir para cada punto
    global_features = layers.Conv1D(1024, 1, activation='relu')(global_features)

    # Concatenar características globales y locales
    x = layers.concatenate([x, global_features])

    # Capa convolucional final para segmentación
    x = layers.Conv1D(512, 1, activation='relu')(x)
    x = layers.Conv1D(256, 1, activation='relu')(x)
    outputs = layers.Conv1D(num_classes, 1, activation='softmax')(x)  # Clasificación por punto

    return tf.keras.Model(inputs, outputs)

# Construir y compilar el modelo
pointnet_model = build_pointnet(num_classes=64, input_dim=4)
# probar con SGD tambien
pointnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Resumen del modelo
pointnet_model.summary()

# -----------------------------
# ENTRENAMIENTO DEL MODELO
# -----------------------------
history = pointnet_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,  # Ajustar según los recursos
    batch_size=8,  # Ajustar según la memoria disponible
    verbose=1
)

# -----------------------------
# EVALUACIÓN
# -----------------------------
loss, accuracy = pointnet_model.evaluate(x_val, y_val)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# -----------------------------
# PREDICCIONES
# -----------------------------
# Predicciones punto a punto
predictions = pointnet_model.predict(x_val)
predicted_labels = np.argmax(predictions, axis=-1)  # Etiquetas por punto
print("Predicciones para la primera nube:", predicted_labels[0])
print("Etiquetas reales para la primera nube:", y_val[0])

# Convertir el objeto history a un DataFrame
history_df = pd.DataFrame(history.history)

pointnet_model.save('/Users/felixmaral/Desktop/TFG/modelos/pointnet_model.h5')

import matplotlib.pyplot as plt

# Gráfica de la pérdida (loss)
plt.figure()
plt.plot(history_df['loss'], label='Entrenamiento')
plt.plot(history_df['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Gráfica de la precisión (accuracy)
plt.figure()
plt.plot(history_df['accuracy'], label='Entrenamiento')
plt.plot(history_df['val_accuracy'], label='Validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Precisión')
plt.legend()
plt.show()


    

        










