import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

def align_sim3(source_traj, target_traj):
    # Convert trajectories to Open3D point cloud format
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_traj[:,:3])
    source_pcd.normals = o3d.utility.Vector3dVector(source_traj[:,3:7])
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_traj[:,:3])
    target_pcd.normals = o3d.utility.Vector3dVector(target_traj[:,3:7])

    # Compute Sim3 transformation
    reg = o3d.registration.registration_sim3_kpts(source_pcd, target_pcd, 0.1)
    sim3 = reg.transformation

    # Apply Sim3 transformation to source trajectory
    source_traj_sim3 = np.zeros_like(source_traj)
    source_traj_sim3[:,:3] = (sim3[:3,:3] @ source_traj[:,:3].T + sim3[:3,3].reshape(-1,1)).T
    source_traj_sim3[:,3:7] = source_traj[:,3:7] @ sim3[:3,:3].T

    return source_traj_sim3, sim3

def compare_frame_numbers(a: str, b: str) -> bool:
    # extract numerical part of the key
    frame_num_a = int(a[6:-9])
    frame_num_b = int(b[6:-9])
    return frame_num_a < frame_num_b

def read_trajectory(file_path):
    pose_dict = {}
    with open(file_path) as f:
        j = json.load(f)

    frames = j["frames"]
    for frame in frames:
        file_path = frame["file_path"]
        transform_matrix = np.array(frame["transform_matrix"])
        translation = transform_matrix[:3, 3]
        rotation = R.from_matrix(transform_matrix[:3, :3])
        quat = rotation.as_quat()
        pose_dict[file_path[file_path.rfind('/')+1:]] = (translation, quat)
    
    return pose_dict


def plot_trajectory(colmap_traj, vicon_traj, transformed_vicon2colmap):
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(8, 12))

    # # Plot x, y, and z coordinates of the trajectory in 3D
    # ax4.plot(transformed_vicon2colmap[:, 0], transformed_vicon2colmap[:, 1], transformed_vicon2colmap[:, 2], label='vicon2colmap')
    # ax4.plot(colmap_traj[:, 0], colmap_traj[:, 1], label='Colmap')
    # ax4.plot(vicon_traj[:, 0], vicon_traj[:, 1], label='Vicon')
    # ax4.set_xlabel('X')
    # ax4.set_ylabel('Y')
    # #ax1.set_zlabel('Z')
    # ax4.legend()

    # Plot x and y coordinates of the trajectory in 2D (top-down view)
    ax2.plot(colmap_traj[:, 0], colmap_traj[:, 1], label='Colmap')
    ax2.plot(vicon_traj[:, 0], vicon_traj[:, 1], label='Vicon')
    ax2.plot(transformed_vicon2colmap[:, 0], transformed_vicon2colmap[:, 1], linestyle='--', color='red', label='vicon2colmap', )
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()

    # Plot z coordinate of the trajectory in 2D (height)
    ax3.plot(colmap_traj[:, 2], label='Colmap')
    ax3.plot(vicon_traj[:, 2], label='Vicon')
    ax3.plot(transformed_vicon2colmap[:, 2], linestyle='--', color='red', label='vicon2colmap')
    ax3.set_xlabel('Frame number')
    ax3.set_ylabel('Height')
    ax3.legend()

    plt.show()

def transform2colmap(vicon_trans, vicon_quat):
    t_col2vicon = np.array([-0.3847422434037402, -0.04886832759628837, -0.3880739867654808])
    q_col2vicon = np.array([-0.7081920519989893, -0.7055147796551348, 0.02642035161927524, 0.003857226197874406])
    scale_col2vicon = 0.4665371357393386

    #R_col2vicon = R.from_quat([q_col2vicon[3], q_col2vicon[0], q_col2vicon[1], q_col2vicon[2]])
    R_col2vicon = R.from_quat(q_col2vicon)
    R_CtoG = R.from_quat(vicon_quat)

    #print("R_CtoG: ")
    #print(R_CtoG.as_matrix())
    #print("R_col2vicon: ")
    #print(R_col2vicon.as_matrix())

    p_CinColmap = np.transpose(R_col2vicon.as_matrix()) @ ((np.array(vicon_trans).reshape(3,1) - t_col2vicon.reshape(3,1)) / scale_col2vicon)
    R_cam2colmap = np.transpose(R_col2vicon.as_matrix()) @ R_CtoG.as_matrix()
    q_cam2colmap = R.from_matrix(R_cam2colmap).as_quat()

    return p_CinColmap, q_cam2colmap

pose_vicon = read_trajectory("/media/saimouli/RPNG_FLASH_4/datasets/ar_table/table_01_openvins_gt/transforms.json")
pose_colmap = read_trajectory("/media/saimouli/RPNG_FLASH_4/datasets/ar_table/table_01_colmap_compare/transforms.json")

img_keys = []
for key in pose_vicon.keys():
    img_keys.append(key)

img_keys = sorted(img_keys, key=lambda x: int(x.split("_")[1].split(".")[0]))
#img_keys.sort(key=compare_frame_numbers)
colmap_traj = []
vicon_traj = []

trans_colmap_traj = []

for key in img_keys:
    if key in pose_colmap and key in pose_vicon:
        col_trans = pose_colmap[key][0].tolist()
        col_quat = pose_colmap[key][1].tolist()
        colmap_traj.append(col_trans + col_quat)
        
        vicon_trans = pose_vicon[key][0].tolist()
        vicon_quat = pose_vicon[key][1].tolist()
        vicon_traj.append(vicon_trans + vicon_quat)

        #TODO: convert vins to colmap and store into the traj and plot 
        p_CinColmap, q_cam2colmap =  transform2colmap(vicon_trans, vicon_quat)
        trans_colmap_traj.append(p_CinColmap.ravel().tolist() + q_cam2colmap.ravel().tolist())


plot_trajectory(np.array(colmap_traj), np.array(vicon_traj), np.array(trans_colmap_traj))
