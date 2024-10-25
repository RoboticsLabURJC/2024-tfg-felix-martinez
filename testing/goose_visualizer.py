import open3d as o3d
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# ------ Global Variables -------

path = '/Users/felixmaral/Desktop/TFG/datasets/goose_3d_val/lidar/val/2023-05-15_neubiberg_rain'
#path = '/home/felix/Escritorio/TFG/datasets/Goose/goose_3d_val/lidar/val/2023-05-15_neubiberg_rain'
colormaps_list = ['plasma', 'jet', 'inferno', 'viridis', 'cividis', 'turbo', 'coolwarm']
reduced = [False]
FPS = [0.5]
zoom_third_person = 0.01
zoom_top = 0.06

# ------- Functions -------

def read_bin_file(file_path):
    '''
    Reads a .bin file and returns an o3d.PointCloud object
    '''
    # Reading a .bin file
    scan = np.fromfile(file_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    # Put in attribute
    points = scan[:, 0:3]       # get xyz
    remissions = scan[:, 3]     # get remission

    return points, remissions

def load_path_bin_files(path) -> list:
    '''
    Loads into an array the names of all existing .bin files in the path and orders them based on the sample number in the format __XXXX
    '''
    def extract_sample_number(file_name):
        # Split the filename by "__" and take the part after "__"
        parts = file_name.split("__")
        if len(parts) > 1:
            # Try to extract the number after the first "__"
            num_str = ''.join([char for char in parts[1] if char.isdigit()])
            return int(num_str) if num_str.isdigit() else 0
        return 0

    # Get the list of all .bin files in the directory
    file_paths = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".bin")]

    # Sort the files based on the sample number extracted by the function
    sorted_file_paths = sorted(file_paths, key=lambda file: extract_sample_number(os.path.basename(file)))

    return sorted_file_paths

def set_colors(remissions, colormap_name='plasma') -> np.ndarray:
    '''
    Normalizes the remissions and applies the specified colormap for visualization.
    '''
    # Normalizing remissions
    norm_remissions = np.asarray(remissions)
    norm_remissions = (norm_remissions - norm_remissions.min()) / (norm_remissions.max() - norm_remissions.min())
    # Setting the specified colormap
    cmap = plt.get_cmap(colormap_name)
    colors = cmap(norm_remissions)[:, :3]  # Get RGB components

    return colors

def update_colors(point_cloud, remissions, colormap_name='plasma') -> None:
    '''
    Updates the point cloud colors using the specified colormap.
    '''
    colors = set_colors(remissions, colormap_name)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

def print_num_points(point_cloud) -> None:
    '''
    Prints in the terminal the number of points of the sample
    '''
    num_points = np.asarray(point_cloud.points).shape[0]
    print(f'Number of points: {num_points}')

def add_new_sample(point_cloud, points, colors) -> None:
    '''
    Adds the new sample to o3d.PointCloud
    '''
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

def update_pointcloud(path_file, point_cloud) -> None:
    '''
    Updates o3d.PointCloud object from a new sample
    '''
    points, remissions = read_bin_file(path_file)
    colors = set_colors(remissions)
    add_new_sample(point_cloud, points, colors)

    print(f'Archivo: {path_file}')

def configure_render_options(vis) -> None:
    '''
    Configures the size of the points
    '''
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1.5
    vis.get_render_option().show_coordinate_frame = False

def configure_camera_third_person(vis) -> None:
    '''
    Configures the camera to view in third person
    '''
    view_control = vis.get_view_control()
    view_control.set_front([-1,0,0.4])
    view_control.set_lookat([0,0,0])
    view_control.set_up([0,0,1])
    view_control.set_zoom(zoom_third_person)

def configure_camera_top(vis) -> None:
    '''
    Configures the camera to view in a top view
    '''
    view_control = vis.get_view_control()
    view_control.set_front([0,0,1])
    view_control.set_lookat([0,0,0])
    view_control.set_up([1,0,0])
    view_control.set_zoom(zoom_top)

def add_sensor_geometry(vis) -> None:
    '''
    Adds an cylindre in the LiDARs position
    '''
    sensor = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3, height=1.0) # geometry
    sensor.translate([0, 0, 0]) # position
    sensor.paint_uniform_color([1, 0, 0]) # color
    vis.add_geometry(sensor)

def add_axis(vis) -> None:
    '''
    Adds the axis into the visualizer
    '''
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])  # Tamaño ajustable
    vis.add_geometry(axis)

def vis_pointcloud(vis, point_cloud) -> None:
    '''
    Visualizes a o3d.PointCloud
    '''
    # clearing previous geometries
    vis.clear_geometries()
    vis.add_geometry(point_cloud)
    add_sensor_geometry(vis)
    add_axis(vis)


# ------ Testing Functions ------

def print_points_first_file() -> None:
    '''
    Prints the number of points from the first file in the path
    '''
    path_file = load_path_bin_files(path)[0]
    points, remissions = read_bin_file(path_file)
    colors = set_colors(remissions)
    point_cloud = o3d.geometry.PointCloud()
    add_new_sample(point_cloud, points, colors)

    print_num_points(point_cloud)

def vis_first_file() -> None:
    '''
    Visualizes the first file in the path
    '''
    path_file = load_path_bin_files(path)[0]
    # creating visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='First Sample')
    point_cloud = o3d.geometry.PointCloud()
    update_pointcloud(path_file, point_cloud)
    vis.add_geometry(point_cloud)

    vis.run()
    vis.destroy_window()

def vis_sequences():
    '''
    Displays sequences of samples at a certains FPS,
    it also has 2 cameras available and the possibility of changing the colormap
    '''
    path_file_list = load_path_bin_files(path)
    num_files = len(path_file_list)
    point_cloud = o3d.geometry.PointCloud()
    points, remissions_data = read_bin_file(path_file_list[0])
    remissions = [remissions_data]  # Hacer remissions mutable usando una lista
    add_new_sample(point_cloud, points, set_colors(remissions[0], colormaps_list[0]))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='PointCloud Sequence')
    vis.add_geometry(point_cloud)
    configure_render_options(vis)
    add_sensor_geometry(vis)
    add_axis(vis)

    is_third_person = [True]
    colormap_index = [0]
    current_colormap = [colormaps_list[0]]
    background = [False]
    is_paused = [False]

    # Configuración inicial de la cámara
    configure_camera_third_person(vis)

    def toggle_camera(vis):
        if is_third_person[0]:
            configure_camera_top(vis)
        else:
            configure_camera_third_person(vis)
        is_third_person[0] = not is_third_person[0]
        vis.update_renderer()

    def toggle_colormap(vis):
        colormap_index[0] = (colormap_index[0] + 1) % len(colormaps_list)
        current_colormap[0] = colormaps_list[colormap_index[0]]
        update_colors(point_cloud, remissions[0], current_colormap[0])
        vis.update_geometry(point_cloud)
        vis.update_renderer()
        print(f'Colormap changed to: {current_colormap[0]}')

    def toggle_background(vis):
        if background[0]:
            vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        else:
            vis.get_render_option().background_color = [0.95, 0.95, 0.95]
        background[0] = not background[0]
        vis.update_renderer()

    def toggle_pause(vis):
        is_paused[0] = not is_paused[0]
        print("Paused" if is_paused[0] else "Playing")

    def increase_fps(vis):
        FPS[0] += 0.5
        print(f"FPS increased to: {FPS[0]}")

    def decrease_fps(vis):
        FPS[0] = max(0.1, FPS[0] - 0.5)  # No bajar de 0.1
        print(f"FPS decreased to: {FPS[0]}")

    # callbacks
    vis.register_key_callback(ord("V"), toggle_camera)  # camera 'V'
    vis.register_key_callback(ord("C"), toggle_colormap)  # colormap 'C'
    vis.register_key_callback(ord("B"), toggle_background)  # background color 'B'
    vis.register_key_callback(32, toggle_pause)  # pausa/reanuda con barra espaciadora
    vis.register_key_callback(ord(","), decrease_fps)  # disminuir FPS con ','
    vis.register_key_callback(ord("."), increase_fps)  # aumentar FPS con '.'

    frame = [0]
    last_update_time = [time.time()]  # Track the time of the last update

    def update_frame(vis):
        if is_paused[0]:
            return

        current_time = time.time()
        if current_time - last_update_time[0] >= 1/FPS[0]:  # Ajusta la velocidad de actualización con FPS
            frame[0] += 1
            if frame[0] >= num_files:
                frame[0] = 0  # Reinicia la secuencia

            points, remissions_data = read_bin_file(path_file_list[frame[0]])
            remissions[0] = remissions_data  # Actualizar remissions para el nuevo archivo
            add_new_sample(point_cloud, points, set_colors(remissions[0], current_colormap[0]))
            vis.update_geometry(point_cloud)
            vis.update_renderer()

            last_update_time[0] = current_time  # Actualiza el tiempo de la última actualización

    # Registrar la actualización de fotogramas
    vis.register_animation_callback(lambda vis: update_frame(vis))

    vis.run()
    vis.destroy_window()


# ------ Main Program ------

def main():
    pass

if __name__ == "__main__":
    
    vis_sequences()
