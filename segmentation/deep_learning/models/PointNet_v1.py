import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from pathlib import Path
import os
import tkinter as tk



# Crear una ventana oculta de Tkinter
from tkinter import filedialog

# Create a hidden Tkinter window
root = tk.Tk()
root.withdraw()  # Hide the main window

# Initial path
initial_path = Path(r'C:\Users\fmartinez\Desktop\reco\datasets_slipt')

try:
    # Select the first path
    print("Select the path for 'x_train'")
    x_train_path = Path(filedialog.askdirectory(
        title="Select the folder for 'x_train'",
        initialdir=initial_path
    ))

    # Select the second path
    print("Select the path for 'y_train'")
    y_train_path = Path(filedialog.askdirectory(
        title="Select the folder for 'y_train'",
        initialdir=initial_path
    ))

    # Select the third path
    print("Select the path for 'x_val'")
    x_val_path = Path(filedialog.askdirectory(
        title="Select the folder for 'x_val'",
        initialdir=initial_path
    ))

    # Select the fourth path
    print("Select the path for 'y_val'")
    y_val_path = Path(filedialog.askdirectory(
        title="Select the folder for 'y_val'",
        initialdir=initial_path
    ))

    # Show the selected paths
    print(f"Path selected for x_train: {x_train_path}")
    print(f"Path selected for y_train: {y_train_path}")
    print(f"Path selected for x_val: {x_val_path}")
    print(f"Path selected for y_val: {y_val_path}")

except Exception as e:
    print(f"Error selecting paths: {e}")

finally:
    root.destroy()  # Ensure the Tkinter window is closed



import pandas as pd
from pathlib import Path

def load_data(x_path, y_path):
    X_data, Y_data = [], []

    for x_file in Path(x_path).iterdir():
        if x_file.is_file() and x_file.suffix == '.csv':
            y_file = Path(y_path) / x_file.name.replace('x_', 'y_')

            # Load x and y
            x_df = pd.read_csv(x_file, header=None, skiprows=1)  # Omitir la primera fila si es encabezado
            y_df = pd.read_csv(y_file, header=None, skiprows=1)  # Omitir la primera fila si es encabezado

            try:
                # Convert to float and select x, y, z features
                x_features = x_df.iloc[:, [0, 1, 2]].astype(float).values
                labels = y_df.iloc[:, 0].values

                X_data.append(x_features)
                Y_data.append(labels)
            except ValueError:
                print(f"Error en el archivo: {x_file}, revisa que los datos sean numéricos.")

    return X_data, Y_data

# Load training and validation data
x_train, y_train = load_data(x_train_path, y_train_path)
x_val, y_val = load_data(x_val_path, y_val_path)




num_classes = 64

# Calculate class weights
def calculate_class_weights(y_data):
    all_labels = np.concatenate(y_data)
    class_counts = np.bincount(all_labels, minlength=num_classes)
    total_samples = len(all_labels)
    class_weights = {i: total_samples / (num_classes * count) for i, count in enumerate(class_counts) if count > 0}
    return class_weights

class_weights = calculate_class_weights(y_train)
print(f"Calculated class weights: {class_weights}")



# Define custom layer for dynamic tiling of global features
class DynamicTileGlobalFeature(layers.Layer):
    def call(self, inputs):
        global_feature, input_tensor = inputs
        num_points = tf.shape(input_tensor)[1]  # Get the number of points dynamically
        global_feature = tf.expand_dims(global_feature, axis=1)  # Add a dimension
        global_feature = tf.tile(global_feature, [1, num_points, 1])  # Repeat to match input points
        return global_feature

# Define PointNet model
def tnet(inputs, k):
    x = layers.Conv1D(64, kernel_size=1, activation='relu')(inputs)
    x = layers.Conv1D(128, kernel_size=1, activation='relu')(x)
    x = layers.Conv1D(1024, kernel_size=1, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)

    # Output transformation matrix
    x = layers.Dense(k * k, 
                    kernel_initializer=tf.keras.initializers.Zeros(), 
                    bias_initializer=tf.keras.initializers.Constant(np.eye(k).flatten()))(x)
    return layers.Reshape((k, k))(x)

def pointnet_segmentation(num_points, num_classes):
    inputs = layers.Input(shape=(None, 3))  # Input shape (N, 3)

    tnet_input = tnet(inputs, 3)
    transformed_inputs = layers.Dot(axes=(2, 1))([inputs, tnet_input])

    x = layers.Conv1D(64, kernel_size=1, activation='relu')(transformed_inputs)
    x = layers.Conv1D(64, kernel_size=1, activation='relu')(x)

    tnet_feature = tnet(x, 64)
    transformed_features = layers.Dot(axes=(2, 1))([x, tnet_feature])

    x = layers.Conv1D(64, kernel_size=1, activation='relu')(transformed_features)
    x = layers.Conv1D(128, kernel_size=1, activation='relu')(x)
    x = layers.Conv1D(1024, kernel_size=1, activation='relu')(x)

    global_feature = layers.GlobalMaxPooling1D()(x)
    global_feature = DynamicTileGlobalFeature()([global_feature, inputs])  # Use the custom layer

    # Segmentation MLP
    concat_features = layers.Concatenate()([transformed_features, global_feature])
    x = layers.Conv1D(512, kernel_size=1, activation='relu')(concat_features)
    x = layers.Conv1D(256, kernel_size=1, activation='relu')(x)
    x = layers.Conv1D(128, kernel_size=1, activation='relu')(x)
    outputs = layers.Conv1D(num_classes, kernel_size=1, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)



num_classes = 64
model = pointnet_segmentation(num_points=None, num_classes=num_classes)
model.summary()

# Compile the model with custom metrics
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'mae', tf.keras.metrics.Precision(name='precision')])




# Train the model
batch_size = 16
epochs = 10

# Train the model with the callback
history = model.fit(
    steps_per_epoch=len(x_train) // batch_size,
    validation_steps=len(x_val) // batch_size,
    epochs=epochs,
)

# Plot training progress
import matplotlib.pyplot as plt

def plot_training_progress(history):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_progress(history)