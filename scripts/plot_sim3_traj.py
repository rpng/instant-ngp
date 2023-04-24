import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    ax2.plot(colmap_traj[:, 0], colmap_traj[:, 1], label='Colmap c2w', linestyle=':')
    ax2.plot(vicon_traj[:, 0], vicon_traj[:, 1], label='Vicon c2w')
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

# hamiltonian matrix
def quat2rot(x, y, z, w):
    rot_mat = np.zeros((3, 3))
    rot_mat[0, 0] = 1 - 2*y**2 - 2*z**2
    rot_mat[0, 1] = 2*x*y - 2*z*w
    rot_mat[0, 2] = 2*x*z + 2*y*w
    rot_mat[1, 0] = 2*x*y + 2*z*w
    rot_mat[1, 1] = 1 - 2*x**2 - 2*z**2
    rot_mat[1, 2] = 2*y*z - 2*x*w
    rot_mat[2, 0] = 2*x*z - 2*y*w
    rot_mat[2, 1] = 2*y*z + 2*x*w
    rot_mat[2, 2] = 1 - 2*x**2 - 2*y**2
    return rot_mat

def write2file(colmap_traj, transformed_vicon2colmap):
    # store colmap traj and vicon2colmap traj into two file
    filename_colmap = '/media/saimouli/RPNG_FLASH_4/datasets/ar_table/results/traj/colmap_traj.txt'
    filename_vicon2colmap = '/media/saimouli/RPNG_FLASH_4/datasets/ar_table/results/traj/vicon2colmap_traj.txt'
    with open(filename_colmap, 'w') as f:
        f.write("# timestamp(s) tx ty tz qx qy qz qw Pr11 Pr12 Pr13 Pr22 Pr23 Pr33 Pt11 Pt12 Pt13 Pt22 Pt23 Pt33\n")
        for i in range(len(colmap_traj)):
            timestamp = i
            pose = colmap_traj[i]
            row = [timestamp] + list(pose[0:3]) + list(pose[3:]) + list(np.zeros(12))
            row_str = " ".join([str(x) for x in row]) + "\n"
            f.write(row_str)
    with open(filename_vicon2colmap, 'w') as f:
        f.write("# timestamp(s) tx ty tz qx qy qz qw Pr11 Pr12 Pr13 Pr22 Pr23 Pr33 Pt11 Pt12 Pt13 Pt22 Pt23 Pt33\n")
        for i in range(len(transformed_vicon2colmap)):
            timestamp = i
            pose = transformed_vicon2colmap[i]
            row = [timestamp] + list(pose[0:3]) + list(pose[3:]) + list(np.zeros(12))
            row_str = " ".join([str(x) for x in row]) + "\n"
            f.write(row_str)

def transform2colmap(vicon_trans, vicon_quat):
    p_MinG = np.array([0.1202150278123188, -0.2416818268398295 ,1.409115977446361])
    q_col2vicon = np.array([0.08893966587659595, -0.4739001129231136, -0.01060197614235877, 0.8760114251007783])
    S_GtoM = 1/0.4105941495512237
    #x,y,z,w format
    R_GtoM = R.from_quat([q_col2vicon[0], q_col2vicon[1], q_col2vicon[2], q_col2vicon[3]])
    #R_col2vicon = R.from_quat(q_col2vicon)
    #R_CtoG = R.from_quat(vicon_quat)
    R_CtoG = R.from_quat([vicon_quat[0], vicon_quat[1], vicon_quat[2], vicon_quat[3]])
    p_CinG = np.array(vicon_trans).reshape(3,1)
    #vicon_cam2wld = -R_GtoC.transpose() @ vicon_trans 

    p_CinM = (R_GtoM.as_matrix()) @ ((p_CinG - p_MinG.reshape(3,1)) * S_GtoM)

    R_cam2colmap = R_GtoM.as_matrix() @ R_CtoG.as_matrix()
    q_cam2colmap = R.from_matrix(R_cam2colmap).as_quat()

    return p_CinM, q_cam2colmap

pose_vicon = read_trajectory("/media/saimouli/RPNG_FLASH_4/datasets/ar_table/table_01_openvins_gt/transforms.json") #read_trajectory("/media/saimouli/RPNG_FLASH_4/datasets/ar_table/table_01_openvins_gt/transforms.json")
pose_colmap = read_trajectory("/media/saimouli/RPNG_FLASH_4/datasets/ar_table/table_01_colmap/transforms.json")

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

        #convert vins to colmap and store into the traj and plot 
        p_CinColmap, q_cam2colmap =  transform2colmap(vicon_trans, vicon_quat)
        trans_colmap_traj.append(p_CinColmap.ravel().tolist() + q_cam2colmap.ravel().tolist())

write2file(np.array(colmap_traj), np.array(trans_colmap_traj))
plot_trajectory(np.array(colmap_traj), np.array(vicon_traj), np.array(trans_colmap_traj))

# # test from openvins imu to ca2colmap

# t_vicon = np.array([2.043300045139766, 0.5857969124660941, 1.1031512650222695])
# R_vicon = np.array([[-0.5058920067510982, 0.2927862900254166, -0.8113864658033059],
#             [0.8517286866776609, 0.020709478457268622, -0.5235727414018592],
#             [-0.1364919945077367, -0.9559531120895893, -0.2598506483430542]])

# t_col2vicon = np.array([-0.02466933839056411, -0.3813245901568731 ,0.4045285497581186])
# q_col2vicon = np.array([-0.01639194085120911, 0.02351608445786384, -0.005617303622487707, 0.9995732809288741])
# scale_col2vicon = 0.4608682590319365

# R_vicon2col = R.from_quat([q_col2vicon[0], q_col2vicon[1], q_col2vicon[2], q_col2vicon[3]])
# vicon_trans = np.array(t_vicon).reshape(3,1)

# p_CinColmap = (R_vicon2col.as_matrix()) @ ((vicon_trans - t_col2vicon.reshape(3,1)) / scale_col2vicon)
# R_cam2colmap = R_vicon2col.as_matrix() @ R_vicon

# print("p_CinColmap: ")
# print(p_CinColmap)
# print("R_cam2colmap: ")
# print(R_cam2colmap)