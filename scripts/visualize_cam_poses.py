import json
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def parse_colmap(file_path):
    with open(file_path, 'r') as f:
        # Skip 4 lines
        for i in range(4):
            f.readline()
        
        # Initialize list to store camera poses
        camera_poses = []
        
        # Read data line by line
        line = f.readline()
        while line:
            data = line.strip().split(', ')
            data = data[0].split(' ')
            qw, qx, qy, qz, tx, ty, tz = map(float, [data[1], data[2], data[3], data[4], data[5], data[6], data[7]])
            # Calculate rotation matrix using quaternion
            R = np.array([[1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                         [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
                         [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy]])
            t = np.array([tx, ty, tz]).reshape(-1, 1)
            # Concatenate rotation matrix and translation vector to form 4x4 matrix
            pose = np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])))
            camera_poses.append(pose)
            
            line = f.readline()
        
        return camera_poses
    
# from: https://github.com/demul/extrinsic2pyramid/blob/main/demo1.py
class CameraParameterLoader:
    def __init__(self):
        print('initialize camera parameter lodaer')

    def get_intrinsic(self, path):
        with open(os.path.join(path, '_camera_settings.json'), 'r') as f:
            param_cam = json.load(f)
            param_intrinsic = param_cam['camera_settings'][0]['intrinsic_settings']
            cx = param_intrinsic['cx']
            cy = param_intrinsic['cy']
            fx = param_intrinsic['fx']
            fy = param_intrinsic['fy']
            s = param_intrinsic['s']
            mat_intrinsic = np.array([[fx, s, cx],
                                      [0, fy, cy],
                                      [0, 0, 1]])
        return mat_intrinsic

    def get_extrinsic(self, path):
        with open(path, 'r') as f:
            param_cam = json.load(f)['camera_data']
            param_translation = param_cam['location_worldframe']
            param_rotation = param_cam['quaternion_xyzw_worldframe']

            mat_rotation = np.quaternion.as_rotation_matrix(
                np.quaternion(param_rotation[3], param_rotation[0], param_rotation[1], param_rotation[2]))
            mat_translation = np.array([[param_translation[0]], [param_translation[1]], [param_translation[2]]])
            mat_extrinsic = np.concatenate(
                [np.concatenate([mat_rotation, mat_translation], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
            return mat_extrinsic


class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()

if __name__ == '__main__':
    # argument : the minimum/maximum value of x, y, z
    visualizer = CameraPoseVisualizer([-5, 5], [-5, 5], [0, 5])

    poses = parse_colmap("/media/saimouli/RPNG_FLASH_4/datasets/nerf_dataset/fern/sparse/0/images.txt")

    # argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
    for pose in poses:
        visualizer.extrinsic2pyramid(pose, 'c', 1)

    visualizer.show()