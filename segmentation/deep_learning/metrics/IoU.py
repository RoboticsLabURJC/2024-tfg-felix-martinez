import numpy as np
import tensorflow as tf

def compute_iou(y_true, y_pred, num_classes):
    """
    Calcula la Intersección sobre la Unión (IoU) para segmentación semántica de nubes de puntos.

    Parámetros:
    - y_true: Tensor con las etiquetas reales, de forma (batch_size, num_points).
    - y_pred: Tensor con las predicciones del modelo (valores de probabilidad), de forma (batch_size, num_points, num_classes).
    - num_classes: Número total de clases.

    Retorna:
    - iou_per_class: IoU por cada clase.
    - mean_iou: Promedio de IoU sobre todas las clases presentes en los datos.
    """
    y_pred_classes = tf.argmax(y_pred, axis=-1)  # Convertir probabilidades en etiquetas
    iou_per_class = []
    
    for c in range(num_classes):
        true_mask = tf.equal(y_true, c)  # Máscara de puntos de clase 'c' en la verdad
        pred_mask = tf.equal(y_pred_classes, c)  # Máscara de predicción de clase 'c'
        
        intersection = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, pred_mask), tf.float32))
        union = tf.reduce_sum(tf.cast(tf.logical_or(true_mask, pred_mask), tf.float32))
        
        if union == 0:  # Evitar división por cero
            iou = tf.constant(0.0)
        else:
            iou = intersection / union
        
        iou_per_class.append(iou)
    
    mean_iou = tf.reduce_mean(tf.stack(iou_per_class))  # IoU promedio sobre clases

    return iou_per_class, mean_iou

# Ejemplo de uso con batch de nubes de puntos:
num_classes = 64  # Según tu dataset GOOSE
batch_size = 8
num_points = 14000

# Simulación de etiquetas y predicciones
y_true_example = tf.random.uniform(shape=(batch_size, num_points), minval=0, maxval=num_classes, dtype=tf.int32)
y_pred_example = tf.random.uniform(shape=(batch_size, num_points, num_classes), minval=0, maxval=1, dtype=tf.float32)
y_pred_example = tf.nn.softmax(y_pred_example, axis=-1)  # Simulación de probabilidades softmax

iou_per_class, mean_iou = compute_iou(y_true_example, y_pred_example, num_classes)
print(f"IoU por clase: {iou_per_class}")
print(f"mIoU (Mean IoU): {mean_iou.numpy()}")
