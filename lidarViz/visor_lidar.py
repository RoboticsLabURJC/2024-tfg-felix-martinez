import dash
from dash import dcc, html
from dash.dependencies import Output, Input, State
import plotly.graph_objs as go
import pandas as pd

# Inicializamos la aplicación Dash
app = dash.Dash(__name__)

# Layout de la aplicación con botones "Siguiente" y "Atrás"
app.layout = html.Div([
    html.H1("Visor de Nube de Puntos LiDAR", style={'textAlign': 'center'}),
    
    # dcc.Graph con tamaño de canvas fijo
    dcc.Graph(
        id='graph-lidar', 
        style={'width': '80vw', 'height': '80vh'}  # Canvas grande y fijo (80% del viewport)
    ),
    
    html.Div([
        html.Button('Atrás', id='btn-atras', n_clicks=0, style={'font-size': '20px'}),
        html.Button('Siguiente', id='btn-siguiente', n_clicks=0, style={'font-size': '20px'}),
    ], style={'textAlign': 'center', 'margin-top': '20px'}),
    
    html.Div(id='output-container', children='Visualizando el archivo 1', style={'textAlign': 'center', 'font-size': '18px'}),

    # Almacena el índice actual en el estado de la sesión
    dcc.Store(id='index-datos', data=0)
])

# Preprocesa los datos LiDAR y prepara los objetos gráficos para optimizar la renderización
def generar_scatter_3d(points, remissions):
    """
    Genera una traza 3D para la visualización de los datos LiDAR usando go.Scatter3d.
    """
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=1,  # Ajustar el tamaño de los puntos
            color=remissions,  # Color basado en las remisiones
            colorscale='Viridis',
            opacity=0.8
        )
    )

# Callback para actualizar el gráfico según el índice de archivo seleccionado
@app.callback(
    [Output('graph-lidar', 'figure'),
     Output('output-container', 'children'),
     Output('index-datos', 'data')],
    [Input('btn-siguiente', 'n_clicks'),
     Input('btn-atras', 'n_clicks')],
    [State('index-datos', 'data')]
)
def actualizar_grafico(n_clicks_siguiente, n_clicks_atras, index_actual):
    """
    Actualiza el gráfico con los datos del archivo LiDAR según el botón que se presione.
    """
    max_index = len(datos_lidar) - 1  # Último índice de los datos disponibles

    # Determinar el nuevo índice basado en los clics de los botones
    if n_clicks_siguiente > n_clicks_atras and index_actual < max_index:
        index_actual += 1
    elif n_clicks_atras > n_clicks_siguiente and index_actual > 0:
        index_actual -= 1

    # Evitar valores negativos o mayores que el número de archivos
    index_actual = max(0, min(index_actual, max_index))

    # Obtener los puntos y remisiones para el índice actual
    points, remissions = datos_lidar[index_actual]

    # Crear la figura con go.Scatter3d
    fig = go.Figure(data=[generar_scatter_3d(points, remissions)])

    # Ajustar el layout de la escena (zoom, ejes)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[points[:, 0].min() - 10, points[:, 0].max() + 10]),
            yaxis=dict(range=[points[:, 1].min() - 10, points[:, 1].max() + 10]),
            zaxis=dict(range=[points[:, 2].min() - 10, points[:, 2].max() + 10])
        ),
        title=f"Nube de Puntos - Archivo {index_actual + 1}"
    )

    # Actualizar el texto que muestra el índice del archivo actual
    output_text = f"Visualizando el archivo {index_actual + 1} de {len(datos_lidar)}"

    # Devolver el gráfico, el texto actualizado y el nuevo índice
    return fig, output_text, index_actual


# Función para iniciar el visor Dash con los datos LiDAR
def iniciar_visor(datos_lidar_procesados):
    """
    Inicia el servidor Dash para visualizar los datos de LiDAR.
    :param datos_lidar_procesados: Lista de tuplas con puntos y remisiones procesadas
    """
    global datos_lidar
    datos_lidar = datos_lidar_procesados  # Asignamos los datos procesados a la variable global

    # Ejecuta el servidor Dash
    app.run_server(debug=True, use_reloader=False)
