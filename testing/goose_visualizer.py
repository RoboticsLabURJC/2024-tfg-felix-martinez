import open3d as o3d
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys
from plyfile import PlyData

# ------ Global Variables -------

#path = '/Users/felixmaral/Desktop/TFG/datasets/goose_3d_val/lidar/val/2023-05-15_neubiberg_rain'
#path = '/Users/felixmaral/Desktop/TFG/datasets/Rellis_3D_lidar_example/os1_cloud_node_color_ply'

#path = '/home/felix/Escritorio/TFG/datasets/Rellis_3D_os1_cloud_node_color_ply/Rellis-3D/00001'
path = '/home/felix/Escritorio/TFG/datasets/Goose/goose_3d_val/lidar/val/2023-01-20_aying_mangfall_2'

colormaps_list = ['plasma', 'jet', 'inferno', 'viridis', 'cividis', 'turbo', 'coolwarm']
FPS = [0.5]
zoom_third_person = 0.01
zoom_top = 0.06

# ------- Functions -------

def read_bin_file(file_path):
    '''
    Reads a .bin file and returns an o3d.PointCloud object.
    Checks if the .bin file exists before attempting to read.
    '''
    if not file_path.endswith('.bin') or not os.path.exists(file_path):
        print(f"Error: .bin file not found at {file_path}")
        print("Please check that the file exists and try again.")
        sys.exit(1)

    scan = np.fromfile(file_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, 0:3]       # get xyz
    remissions = scan[:, 3]     # get remission

    return points, remissions

def read_ply_file(file_path):
    '''
     Reads a .ply file and returns an o3d.PointCloud object.
    Checks if the .ply file exists before attempting to read.
    '''
    if not file_path.endswith('.ply') or not os.path.exists(file_path):
        print(f"Error: .bin file not found at {file_path}")
        print("Please check that the file exists and try again.")
        sys.exit(1)
    
    plydata = PlyData.read(file_path)

    # components
    x = plydata['vertex'].data['x']
    y = plydata['vertex'].data['y']
    z = plydata['vertex'].data['z']
    puntos = np.vstack((x, y, z)).T  # array [n,3]

    # Extraer la intensidad
    intensidad = plydata['vertex'].data['intensity']

    return puntos, intensidad

def load_path_files(path) -> list:
    '''
    Loads the names of all existing .bin and .ply files in the path.
    '''
    def extract_sample_number(file_name):
        parts = file_name.split("__")
        if len(parts) > 1:
            num_str = ''.join([char for char in parts[1] if char.isdigit()])
            return int(num_str) if num_str.isdigit() else 0
        return 0

    file_paths = [os.path.join(path, file) for file in os.listdir(path)]
    sorted_file_paths = sorted(file_paths, key=lambda file: extract_sample_number(os.path.basename(file)))
    return sorted_file_paths

def set_colors(remissions, colormap_name='plasma') -> np.ndarray:
    '''
    Normalizes the remissions and applies the specified colormap for visualization.
    '''
    norm_remissions = np.asarray(remissions)
    norm_remissions = (norm_remissions - norm_remissions.min()) / (norm_remissions.max() - norm_remissions.min())
    cmap = plt.get_cmap(colormap_name)
    colors = cmap(norm_remissions)[:, :3]
    return colors

def update_colors(point_cloud, remissions, colormap_name='plasma') -> None:
    colors = set_colors(remissions, colormap_name)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

def add_new_sample(point_cloud, points, colors) -> None:
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

def resample_points(points, remissions, factor=3):
    '''
    Resamples the point cloud with a 1:factor ratio.
    '''
    indices = np.arange(0, points.shape[0], factor)
    resampled_points = points[indices]
    resampled_remissions = remissions[indices]
    return resampled_points, resampled_remissions

def configure_render_options(vis):
    '''
    Sets parameters for the renderer, including a black background.
    '''
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]  # Black background
    vis.get_render_option().point_size = 1.5

def configure_camera_third_person(vis):
    '''
    Configures the camera to view in third person.
    '''
    view_control = vis.get_view_control()
    view_control.set_front([-1, 0, 0.4])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(zoom_third_person)

def configure_camera_top(vis):
    '''
    Configures the camera to view in a top view.
    '''
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, 1])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([1, 0, 0])
    view_control.set_zoom(zoom_top)

def vis_sequences():
    '''
    Displays sequences of samples at a certain FPS,
    it also has 2 cameras available and the possibility of changing the colormap
    '''

    is_bin_file = [True]

    path_file_list = load_path_files(path)
    bin_files = [file for file in path_file_list]

    if bin_files[0].endswith('.ply'):
        is_bin_file[0] = False

    num_files = len(bin_files)
    point_cloud = o3d.geometry.PointCloud()

    if is_bin_file[0]:
        points, remissions_data = read_bin_file(bin_files[0])
    else:
        points, remissions_data = read_ply_file(bin_files[0])

    remissions = [remissions_data]
    add_new_sample(point_cloud, points, set_colors(remissions[0], colormaps_list[0]))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='PointCloud Sequence')
    configure_render_options(vis)  # Configure black background and point size
    vis.add_geometry(point_cloud)

    configure_camera_third_person(vis)

    is_third_person = [True]
    colormap_index = [0]
    current_colormap = [colormaps_list[0]]
    background = [False]
    is_paused = [False]
    is_auto_mode = [True]
    is_resampled = [False]

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
        FPS[0] = max(0.1, FPS[0] - 0.5)
        print(f"FPS decreased to: {FPS[0]}")

    def toggle_mode(vis):
        is_auto_mode[0] = not is_auto_mode[0]
        mode = "Automatic" if is_auto_mode[0] else "Manual"
        print(f"Mode changed to: {mode}")

    def toggle_resampling(vis):
        is_resampled[0] = not is_resampled[0]
        print("Resampling mode:", "1:3" if is_resampled[0] else "Original")
        update_point_cloud()

    def next_frame():
        frame[0] = (frame[0] + 1) % num_files
        update_point_cloud()

    def prev_frame():
        frame[0] = (frame[0] - 1) % num_files
        update_point_cloud()

    def update_point_cloud():
        if is_bin_file[0]:
            points, remissions_data = read_bin_file(bin_files[frame[0]])
        else:
            points, remissions_data = read_ply_file(bin_files[frame[0]])
        if is_resampled[0]:
            points, remissions_data = resample_points(points, remissions_data, factor=3)
        remissions[0] = remissions_data
        add_new_sample(point_cloud, points, set_colors(remissions[0], current_colormap[0]))
        vis.update_geometry(point_cloud)
        vis.update_renderer()

    vis.register_key_callback(ord("V"), toggle_camera)
    vis.register_key_callback(ord("C"), toggle_colormap)
    vis.register_key_callback(ord("B"), toggle_background)
    vis.register_key_callback(32, toggle_pause)
    vis.register_key_callback(265, increase_fps)
    vis.register_key_callback(264, decrease_fps)
    vis.register_key_callback(ord("M"), toggle_mode)
    vis.register_key_callback(ord("N"), toggle_resampling)
    vis.register_key_callback(262, lambda vis: next_frame() if not is_auto_mode[0] else None)
    vis.register_key_callback(263, lambda vis: prev_frame() if not is_auto_mode[0] else None)

    frame = [0]
    last_update_time = [time.time()]

    def update_frame(vis):
        if is_paused[0] or not is_auto_mode[0]:
            return

        current_time = time.time()
        if current_time - last_update_time[0] >= 1 / FPS[0]:
            next_frame()
            last_update_time[0] = current_time

    vis.register_animation_callback(lambda vis: update_frame(vis))

    vis.run()
    vis.destroy_window()

# ------ Main Program ------

if __name__ == "__main__":
    vis_sequences()