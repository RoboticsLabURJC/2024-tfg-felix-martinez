import open3d as o3d
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from plyfile import PlyData

# ------ Global Variables -------

path = '/home/felix/Escritorio/TFG/datasets/Goose/goose_3d_val/lidar/val/2023-05-15_neubiberg_rain'
colormaps_list = ['plasma', 'jet', 'inferno']
reduced = [False]

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

def load_path_files(path) -> list:
    '''
    Loads into a array the name of all existing files in a path
    '''
    file_paths = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".bin")]

    return file_paths

def set_colors(remissions) -> np.ndarray:
    '''
    Normalizes the remissions and set a colormap for an later linear visualization
    '''
    # Normalizing remissions
    norm_remissions = np.asarray(remissions)
    norm_remissions = (norm_remissions - norm_remissions.min()) / (norm_remissions.max() - norm_remissions.min())
    # Setting a colormap
    cmap = plt.get_cmap(colormaps_list[0])
    colors = cmap(norm_remissions)[:, :3]

    return colors

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

def configure_visualizer(vis) -> None:
    '''
    Configures some parameters of the visualizer
    '''
    view_control = vis.get_view_control()
    view_control.set_front([1,0,0.7])
    view_control.set_lookat([0,0,0])
    view_control.set_up([0,0,1])
    view_control.set_zoom(0.02)

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
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=4.0, origin=[0, 0, 0])  # Tamaño ajustable
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
    path_file = load_path_files(path)[0]
    points, remissions = read_bin_file(path_file)
    colors = set_colors(remissions)
    point_cloud = o3d.geometry.PointCloud()
    add_new_sample(point_cloud, points, colors)

    print_num_points(point_cloud)

def vis_first_file() -> None:
    '''
    Visualizes the first file in the path
    '''
    path_file = load_path_files(path)[0]
    # creating visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='First Sample')
    point_cloud = o3d.geometry.PointCloud()
    update_pointcloud(path_file, point_cloud)
    vis.add_geometry(point_cloud)

    vis.run()
    vis.destroy_window()

def vis_sequences():

    path_file_list = load_path_files(path)
    num_files = len(path_file_list)
    point_cloud = o3d.geometry.PointCloud()
    update_pointcloud(path_file_list[0], point_cloud)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='PointCloud Sequence')
    configure_visualizer(vis)

    vis.add_geometry(point_cloud)

    frame = [0]
    last_update_time = [time.time()]  # Track the time of the last update

    def update_frame(vis):
        current_time = time.time()

        if current_time - last_update_time[0] >= 0.001:
            frame[0] += 1
            if frame[0] >= num_files:
                frame[0] = 0  # Reset to loop the sequence if desired

            update_pointcloud(path_file_list[frame[0]], point_cloud)
            vis.update_geometry(point_cloud)
            vis.poll_events()
            vis.update_renderer()

            last_update_time[0] = current_time  # Update the time of the last update

    # Set a timer callback to update the frame
    vis.register_animation_callback(lambda vis: update_frame(vis))
    
    vis.run()
    vis.destroy_window()

# ------ Main Program ------

def main():
    pass

if __name__ == "__main__":
    
    vis_sequences()
