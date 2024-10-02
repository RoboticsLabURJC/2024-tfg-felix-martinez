import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm

# Clase base para la lectura de datos LiDAR del dataset GOOSE
class LidarDataReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.read_lidar_data_from_bin()

    # Función para leer datos LiDAR desde un archivo .bin del dataset GOOSE
    def read_lidar_data_from_bin(self):
        try:
            # Leer los datos del archivo binario
            scan = np.fromfile(self.file_path, dtype=np.float32)
            # Los datos LiDAR suelen tener 4 columnas: X, Y, Z, Intensidad
            points = scan.reshape((-1, 4))
            return points
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {self.file_path}")
            return np.array([])

# Clase derivada para la visualización de datos LiDAR
class LidarVisualizer(LidarDataReader):
    def __init__(self, file_path):
        # Inicializar la clase base con el archivo de datos LiDAR
        super().__init__(file_path)

    # Función que convierte intensidad a color en formato hexadecimal
    def intensity_to_color(self, intensity):
        # Normalizamos la intensidad
        intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        # Usamos una colormap de matplotlib (ahora correcto)
        colormap = cm.get_cmap('plasma')
        rgba_colors = colormap(intensity_normalized)
        # Convertimos los valores rgba a hex
        hex_colors = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b, _ in rgba_colors]
        return hex_colors

    # Función que genera un gráfico 3D con Plotly
    def plot_lidar_data(self):
        if self.data.size == 0:
            print("No hay datos LiDAR para visualizar.")
            return

        # Separamos los puntos XYZ e intensidades
        x_vals = self.data[:, 0]
        y_vals = self.data[:, 1]
        z_vals = self.data[:, 2]
        intensities = self.data[:, 3]

        # Convertir intensidades a colores
        colors = self.intensity_to_color(intensities)

        # Crear etiquetas de texto para las intensidades
        text_vals = [f'Intensidad: {intensity}' for intensity in intensities]

        # Crear la visualización 3D con Plotly
        fig = go.Figure(data=[go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='markers',
            marker=dict(
                size=0.8,
                color=colors,
                opacity=0.8
            ),
            text=text_vals,  # Mostrar las intensidades en los tooltips
        )])

        # Configuración del diseño
        fig.update_layout(
            scene=dict(
                xaxis_title='Eje X',
                yaxis_title='Eje Y',
                zaxis_title='Eje Z',
            ),
            title="Gráfico 3D de puntos LiDAR del dataset GOOSE"
        )

        # Personalizar ejes
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    showline=True,
                    zeroline=True,
                    showgrid=False,
                    zerolinecolor='black',
                    zerolinewidth=2
                ),
                yaxis=dict(
                    showline=True,
                    zeroline=True,
                    showgrid=False,
                    zerolinecolor='black',
                    zerolinewidth=2
                ),
                zaxis=dict(
                    showline=True,
                    zeroline=True,
                    showgrid=False,
                    zerolinecolor='black',
                    zerolinewidth=2
                )
            )
        )

        # Mostrar el gráfico
        fig.show()

# Especifica la ruta del archivo .bin del GOOSE dataset
file_path = "/home/felix/Escritorio/TFG/datasets/Goose/goose_3d_val/lidar/val/2022-07-22_flight/2022-07-22_flight__0195_1658494596120446131_vls128.bin"
# Crear una instancia de la clase LidarVisualizer y generar la visualización
visualizer = LidarVisualizer(file_path)
visualizer.plot_lidar_data()
