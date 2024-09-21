import numpy as np
import plotly.graph_objects as go

num_points = 10
points_range = 20

# Función que crea pundos LiDAR simulados
def create_point(): 
    # Creación de componentes espaciales y de color
    x = np.random.rand() * points_range - points_range/2
    y = np.random.rand() * points_range - points_range/2
    z = np.random.rand() * points_range - points_range/2
    r = int(round(np.random.rand() * 255))
    g = int(round(np.random.rand() * 255))
    b = int(round(np.random.rand() * 255))
    # Creación del array
    point = np.array([x, y, z, r, g, b])

    return point

# Función para convertir RGB a formato hexadecimal
def rgb_to_hex(r, g, b):
    # Conviertir valores RGB (0-255) a un color en formato hexadecimal del formato #FFFFFF
    return f'#{r:02x}{g:02x}{b:02x}'


# Función que crea una nube de puntos aleatorios
def create_array(n):
    points = []
    for idx in range(n):
        point = create_point()
        points.append(point)
    lidar_array = np.array(points)

    return lidar_array

data = create_array(num_points)

# Función que genera un gráfico 3D con Plotly
def plot_lidar_data(data):
    # Genera un gráfico 3D interactivo usando Plotly a partir de un array de puntos
    # Extraer coordenadas x, y, z del array y las componentes de color R, G, B
    x_vals = data[:, 0]
    y_vals = data[:, 1]
    z_vals = data[:, 2]
    r_vals = data[:, 3]
    g_vals = data[:, 4]
    b_vals = data[:, 5]

    # Aplicamos la transformación a hexadecimal y creamos una lista 'colors' que tendra la componente #FFFFFF de cada punto
    colors = [rgb_to_hex(r, g, b) for r, g, b in zip(r_vals, g_vals, b_vals)]

    # Crear el gráfico 3D
    fig = go.Figure(data=[go.Scatter3d(
        x = x_vals,
        y = y_vals,
        z = z_vals,
        mode = 'markers',
        marker = dict(
            size = 3,
            color = colors,           
            colorscale = 'Viridis',       # Escala de colores
            opacity = 0.8
        )
    )])
    
    # Etiquetar los ejes
    fig.update_layout(
        scene=dict(
            xaxis_title='Eje X',
            yaxis_title='Eje Y',
            zaxis_title='Eje Z'
        ),
        title="Gráfico 3D de puntos"
    )
    
    # Mostrar el gráfico
    fig.write_html("grafico.html")
    fig.show()
    