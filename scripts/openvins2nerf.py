import rosbag
import cv2
import os
import sys
import numpy as np
import time
from cv_bridge import CvBridge
import math
from geometry_msgs.msg import PoseWithCovarianceStamped
import json
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

cam2wld_traj = []
wld2cam_traj = []

class Pose:
    def __init__(self, x, y, z, x_orient, y_orient, z_orient, w_orient):
        self.position = Position(x, y, z)
        self.orientation = Orientation(x_orient, y_orient, z_orient, w_orient)

class Position:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Orientation:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

def ros_to_pose_matrix(pose):
    tvec = np.array([pose.position.x, pose.position.y, pose.position.z])
    qvec = np.array([pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w]) 
    #R_GtoI = jpl_quat_to_rot_mat(qvec[0], qvec[1], qvec[2], qvec[3]) #x,y,z,w
    R_ItoG = quat2rot(qvec[0], qvec[1], qvec[2], qvec[3])
    p_IinG = tvec.reshape([3, 1])
    T_wld2imu = np.hstack((R_ItoG, p_IinG))
    return T_wld2imu

def plot_trajectory():
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(8, 12))
    
    ax2.plot(np.array(cam2wld_traj)[:, 0], np.array(cam2wld_traj)[:, 1], label='cam2wld', linestyle=':')
    ax2.plot(np.array(wld2cam_traj)[:, 0], np.array(wld2cam_traj)[:, 1], label='wld2cam', linestyle='--')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()

    ax3.plot(np.array(cam2wld_traj)[:, 2], label='cam2wld')
    ax3.plot(np.array(wld2cam_traj)[:, 2], label='wld2cam')
    ax3.set_xlabel('m')
    ax3.set_ylabel('z(m)')
    ax3.legend()

    plt.show()

def compute_poses_to_plot(pose):
    R_ItoG = pose[:3, :3]
    p_IinG = pose[:3, 3].reshape([3, 1]) #in reality it is p_IinG

    # rigid transformation from cam2imu
    T_CtoI = np.array([[0.9999654398038452, 0.007342326779113337, -0.003899927610975742, -0.027534314618518095],
              [-0.0073452195116216765, 0.9999727585590525, -0.0007279355223411334, -0.0030587146933711722],
              [0.0038944766308488753, 0.0007565561891287445, 0.9999921303062861, -0.023605118842939803],
              [0.0, 0.0, 0.0, 1.0]])
    
    p_IinC = T_CtoI[:3, 3] #in reality it is p_IinC
    R_ItoC = T_CtoI[:3, :3] #in reality it is R_ItoC

    #adjustments
    p_GinI = -R_ItoG.transpose() @ p_IinG
    R_CtoI = R_ItoC.transpose()

    # transform for cam2wld
    p_GinC = R_ItoC @ p_GinI + p_IinC.reshape([3,1])
    R_CtoG = R_ItoG @ R_CtoI

    # cam2wld
    p_CinG = -R_CtoG @ p_GinC

    cam2wld_traj.append(p_CinG.reshape([3,]).tolist())
    wld2cam_traj.append(p_GinC.reshape([3,]).tolist())

from pyquaternion import Quaternion

# qw_str, qx_str, qy_str, qz_str, tx_str, ty_str, tz_str, camera_id_str, img_name;
#NOTE: Values are hardcoded for rpng table dataset
def write_image_and_pose(image, pose, output_dir, counter, up):
    # Get the timestamp from the pose message
    # timestamp = pose.header.stamp.to_sec()
    # Convert the image message to a cv2 image
    image = CvBridge().imgmsg_to_cv2(image, "bgr8")
    # Construct the file names
    image_file = os.path.join(output_dir, f"frame_{counter}.png")

    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    
    sharp_img = sharpness(image)
    print(image_file, "sharpness=", sharp_img)
    #tvec = np.array([pose.position.x, pose.position.y, pose.position.z])
    #qvec = np.array([pose.orientation.x, pose.orientation.y,
    #                pose.orientation.z, pose.orientation.w])  # x,y,z,w
    R_ItoG = np.array(pose[:3, :3])
    p_IinG = pose[:3, 3].reshape([3, 1]) #in reality it is p_IinG 
 
    # rigid transformation from cam2imu
    T_CtoI = np.array([[0.9999654398038452, 0.007342326779113337, -0.003899927610975742, -0.027534314618518095],
              [-0.0073452195116216765, 0.9999727585590525, -0.0007279355223411334, -0.0030587146933711722],
              [0.0038944766308488753, 0.0007565561891287445, 0.9999921303062861, -0.023605118842939803],
              [0.0, 0.0, 0.0, 1.0]])
    
    p_IinC = T_CtoI[:3, 3] #in reality it is p_IinC
    R_ItoC = T_CtoI[:3, :3] #in reality it is R_ItoC

    #adjustments
    p_GinI = -R_ItoG.transpose() @ p_IinG
    R_CtoI = R_ItoC.transpose()

    # transform for cam2wld
    p_GinC = R_ItoC @ p_GinI + p_IinC.reshape([3,1])
    R_CtoG = R_ItoG @ R_CtoI
    
    p_CinG = -R_CtoG @ p_GinC

    #R_GtoN = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
    #R_CtoN = R_GtoN @ R_CtoG
    c2w = np.concatenate([np.concatenate([R_CtoG, p_CinG], 1), bottom], 0)
    #w2c = np.concatenate([np.concatenate([R_CtoG.transpose(), -R_CtoG.transpose() @ p_GinC], 1), bottom], 0) #wld2cam
    
    # print("t_cam2wld: ", p_GinC)
    # print("R_cam2wld: ", R_CtoG)
    #print("w2c: ", w2c)

    # print("w2c rot: ", w2c)
    #w2c = Rx(np.pi) @ w2c
    # # convert to nerf coordinates
    # c2w[0:3, 2] *= -1  # flip the y and z axis
    # c2w[0:3, 1] *= -1
    # c2w = c2w[[1, 0, 2, 3], :]
    # c2w[2, :] *= -1  # flip whole world upside down
    # print("c2w: ", c2w)

    # rotate the camera to face the table
    up += c2w[0:3, 1]

    frame = {"file_path": image_file, "sharpness": sharp_img, "transform_matrix": c2w}
    

    cv2.imwrite(image_file, image)

    return frame, up

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

# see eq. 92 from quat tech report

def jpl_quat_to_rot_mat(x, y, z, w):
    rot_mat = np.array([
        [1 - 2*y**2 - 2*z**2, 2*(x*y + w*z), 2*(x*z - w*y)],
        [2*(x*y - w*z), 1 - 2*x**2 - 2*z**2, 2*(y*z + w*x)],
        [2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*x**2 - 2*y**2]
    ])
    
    return rot_mat

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

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

def pose_interp(t, t1, t2, aff1, aff2):
    # assume here t1 <= t <= t2
    alpha = 0.0
    if t2 != t1:
        alpha = (t - t1) / (t2 - t1)

    rot_list = R.from_matrix([aff1[:3, :3], aff2[:3, :3]])

    slerp = Slerp([t1,t2], rot_list)
    interp_rots = slerp([t])

    trans1 = aff1[:3, 3]
    trans2 = aff2[:3, 3]

    result = np.eye(4)
    result[:3, 3] = (1.0 - alpha) * trans1 + alpha * trans2
    result[:3, :3] = interp_rots.as_matrix()

    return result

def interpolate_cubic_spline_poses(timestamps, poses):
    # create cubic spline functions for each dimension of the poses
    spline_funcs = [CubicSpline(timestamps, poses[:, i]) for i in range(3)]
    rotation_spline_funcs = [CubicSpline(timestamps, poses[:, i]) for i in range(3, 7)]

    # interpolate the spline functions at desired timestamps to get the desired poses
    desired_timestamps = np.linspace(timestamps[0], timestamps[-1], 900000)
    desired_poses = np.zeros((len(desired_timestamps), 7))
    for i, func in enumerate(spline_funcs):
        desired_poses[:, i] = func(desired_timestamps)
    for i, func in enumerate(rotation_spline_funcs):
        desired_poses[:, i+3] = func(desired_timestamps)
    
    desired_poses_dict = {}
    for i, timestamp in enumerate(desired_timestamps):
        desired_poses_dict[timestamp] = desired_poses[i]
    
    return desired_poses_dict 

def read_bag(bag_file, output_dir, start_time, end_time, out, OUT_PATH, kf_time, pose_topic, interpolate_linear_poses = False, interpolate_cubic_poses_flag = True, gt_file = None):
    first_pose = True
    first_pose_gt = True
    time_temp_gt = 0
    up = np.zeros(3)
    frames_counter = 0
    pose_gt_dict = {}
    timestamps_gt = []; poses_gt = []
    if gt_file is not None: 
        with open(gt_file, "r") as f:
            next(f)
            for line in f: 
                data = line.split()
                timestamps_gt.append(float(data[0]))
                poses_gt.append([float(x) for x in data[1:]])
    
    timestamps_gt = np.array(timestamps_gt)
    poses_gt = np.array(poses_gt)

    if gt_file is not None:
        print("Reading gt file")
        with open(gt_file) as f:
            next(f) # skip first line assuming it will always be a comment 
            for line in f:
                line = line.strip()
                data = line.split(" ")
                timestamp = float(data[0])
                x_pos = float(data[1])
                y_pos = float(data[2])
                z_pos = float(data[3])
                x_orient = float(data[4])
                y_orient = float(data[5])
                z_orient = float(data[6])
                w_orient = float(data[7])
                pose = Pose(x_pos, y_pos, z_pos, x_orient, y_orient, z_orient, w_orient)
                pose_gt_dict[timestamp] = pose
                # if (first_pose_gt == True):
                #     time_temp_gt = timestamp
                #     first_pose_gt = False

                # if timestamp - time_temp_gt >= kf_time:
                #     x_pos = float(data[1])
                #     y_pos = float(data[2])
                #     z_pos = float(data[3])
                #     x_orient = float(data[4])
                #     y_orient = float(data[5])
                #     z_orient = float(data[6])
                #     w_orient = float(data[7])
                #     pose = Pose(x_pos, y_pos, z_pos, x_orient, y_orient, z_orient, w_orient)
                #     pose_gt_dict[timestamp] = pose
                #     time_temp_gt = timestamp

    print("Reading bag, hold tight!")
    time_temp = 0
    first_pose_topic = True
    # Open the ROS bag
    with rosbag.Bag(bag_file, "r") as bag:
        # Create a bridge for converting image messages
        bridge = CvBridge()
        # Initialize the dictionaries for storing image and pose messages
        image_dict = {}
        pose_dict = {}
        
        # Read the messages from the bag
        for topic, msg, t in bag.read_messages():
            if first_pose == True:
                start_pose_time = t.to_sec()
                first_pose = False

            # check if within the provided time range
            if start_time <= t.to_sec() - start_pose_time <= end_time:
                ##print(t.to_sec() - start_pose_time)
                if topic == "/d455/color/image_raw":
                    # Store the image message in the dictionary
                    image_dict[msg.header.stamp.to_sec()] = msg
                elif topic == pose_topic:
                    if first_pose_topic:
                        time_temp = t.to_sec()
                        first_pose_topic = False
                    # storing key frame logic here because poses are higher hz
                    # select a pose for every 1 sec
                    #if t.to_sec() - time_temp >= kf_time:
                        #print(t.to_sec() - time_temp)
                        #time_temp = t.to_sec()
                        # Store the pose message in the dictionary
                        pose_dict[msg.header.stamp.to_sec()] = msg.pose.pose
                        frames_counter+= 1
                    

            # break the loop if the time is out of range
            elif end_time  <= t.to_sec() - start_pose_time:
                break
        
        if gt_file is not None: 
            pose_dict = pose_gt_dict

        print("pose len: ", len(pose_dict))
        print("img len: ", len(image_dict))
        # Iterate over the timestamps
        counter = 0
        
        # # find the nearest timestamp of images based on pose
        # for timestamp in sorted(pose_dict.keys()):
        #     print("pose timestamp: ", timestamp)
        #     time_key = nearest(image_dict.keys(), timestamp)
        #     print("img timestamp: ", time_key)
        #     frame, up = write_image_and_pose(
        #         image_dict[time_key], pose_dict[timestamp], output_dir, counter, up)
        #     counter += 1
        #     out["frames"].append(frame)
        
        #find the nearest timestamp of pose based on images
        pose_keys_sorted = sorted(pose_dict.keys())
        for timestamp in sorted(image_dict.keys()):
            if counter % 7 == 0:
                print("Img timestamp: ", timestamp)
                curr_pose_time = nearest(pose_dict.keys(), timestamp)
                print("Pose timestamp: {} difference: {} ms".format(curr_pose_time, abs(curr_pose_time - timestamp) * 1000))

                # skip if image is very blurry
                if sharpness(bridge.imgmsg_to_cv2(image_dict[timestamp], "bgr8")) > 900:
                    # Get the index of the current timestamp in the sorted list of keys
                    if interpolate_linear_poses == True:
                        index_current = pose_keys_sorted.index(curr_pose_time)

                        if index_current == 0 or index_current == len(pose_keys_sorted) - 1:
                        # closest pose is before the first pose or after the last pose, so skip the image timestamp
                            continue

                        # curr_pose_time <= image_time_stamp <= next_pose_timestamp
                        index_pose_next = index_current + 1
                        next_pose_timestamp = pose_keys_sorted[index_pose_next]
                        print("Current pose time:{} Next timestamp:{}".format(curr_pose_time, next_pose_timestamp))
                        
                        curr_pose = ros_to_pose_matrix(pose_dict[curr_pose_time])
                        next_pose = ros_to_pose_matrix(pose_dict[next_pose_timestamp])

                        # interpolate if current image time is in between poses
                        if curr_pose_time <= timestamp <= next_pose_timestamp:
                            T_inter = pose_interp(timestamp, curr_pose_time, next_pose_timestamp, curr_pose, next_pose)
                            frame, up = write_image_and_pose(image_dict[timestamp], T_inter, output_dir, counter, up)
                        
                        elif pose_keys_sorted[index_current -1] <= timestamp <= pose_keys_sorted[index_current]:
                            print("Going back current pose time:{} Using current timestamp:{} difference: {} ms".format(pose_keys_sorted[index_current -1], \
                                                                                                            pose_keys_sorted[index_current], \
                                                                                                            abs(pose_keys_sorted[index_current -1] - pose_keys_sorted[index_current]) * 1000))
                            curr_pose = ros_to_pose_matrix(pose_dict[pose_keys_sorted[index_current -1]])
                            next_pose = ros_to_pose_matrix(pose_dict[pose_keys_sorted[index_current]])
                            T_inter = pose_interp(timestamp, pose_keys_sorted[index_current -1], pose_keys_sorted[index_current], curr_pose, next_pose)
                            frame, up = write_image_and_pose(image_dict[timestamp], T_inter, output_dir, counter, up)

                        elif pose_keys_sorted[index_current -1] <= timestamp <= next_pose_timestamp:
                            # get one step back and interpolate (this might be inaccurate)
                            print("Going back current pose time:{} Next timestamp:{} difference: {} ms".format(pose_keys_sorted[index_current -1], \
                                                                                                            next_pose_timestamp, \
                                                                                                            abs(pose_keys_sorted[index_current -1] - next_pose_timestamp) * 1000))
                            curr_pose = ros_to_pose_matrix(pose_dict[pose_keys_sorted[index_current -1]])
                            T_inter = pose_interp(timestamp, pose_keys_sorted[index_current -1], next_pose_timestamp, curr_pose, next_pose)
                            frame, up = write_image_and_pose(image_dict[timestamp], T_inter, output_dir, counter, up)
                        
                        else:
                            # else use the closest near time
                            print("Using nearest pose. DID NOT INTERPOLATE")
                            T_inter = curr_pose
                            frame, up = write_image_and_pose(image_dict[timestamp], T_inter, output_dir, counter, up)
                    
                    elif interpolate_cubic_poses_flag == True:
                        desired_pose_dict = interpolate_cubic_spline_poses(timestamps_gt, poses_gt)
                        #print("Cubic interpolation current pose time:{} inter timestamp:{}".format(timestamp, timestamp))
                        if timestamp in desired_pose_dict:
                            desired_pose_at_timestamp = desired_pose_dict[timestamp]
                        else:
                            timestamps = np.array(list(desired_pose_dict.keys()))
                            poses = np.array(list(desired_pose_dict.values()))
                            spline_funcs = [CubicSpline(timestamps, poses[:,i]) for i in range(7)]
                            desired_pose_at_timestamp = np.zeros(7)
                            for i, func in enumerate(spline_funcs):
                                desired_pose_at_timestamp[i] = func(timestamp)
                            translation = desired_pose_at_timestamp[:3]
                            quaternion = desired_pose_at_timestamp[3:]
                            rot = jpl_quat_to_rot_mat(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
                            #rot = quat2rot(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
                            desired_pose_matrix = np.concatenate((rot, np.expand_dims(translation, axis=1)), axis=1)
                            frame, up = write_image_and_pose(image_dict[timestamp], desired_pose_matrix, output_dir, counter, up)
                        print("Cubic interpolation nearest pose:{} inter pose:{}\n".format(ros_to_pose_matrix(pose_dict[curr_pose_time]), desired_pose_matrix))
                        
                    else:
                        # Do not interpolate 
                        print("not interpolating")
                        curr_pose = ros_to_pose_matrix(pose_dict[curr_pose_time])
                        frame, up = write_image_and_pose(image_dict[timestamp], curr_pose, output_dir, counter, up)                        
                        compute_poses_to_plot(curr_pose)
                    out["frames"].append(frame)
                else:
                    print("Skipping blurry img")
            counter += 1
        

        # if gt_file is None:
        #     # find the nearest timestamp of images based on pose 
        #     for timestamp in sorted(pose_dict.keys()):
        #         print("pose timestamp: ", timestamp)
        #         time_key = nearest(image_dict.keys(), timestamp)
        #         print("img timestamp: ", time_key)
        #         counter += 1
        #         # if time_key in image_dict:
        #         #     counter += 1
        #         #     print(counter)
        #         frame, up = write_image_and_pose(
        #             image_dict[time_key], pose_dict[timestamp], output_dir, counter, up)
        #         out["frames"].append(frame)
        # else:
        #     # find the nearest timestamp of pose based on images 
        #     # Use this for GT poses as they are more frequent than images 
        #     for timestamp in sorted(image_dict.keys()):
        #         print("img timestamp: ", timestamp)
        #         time_key = nearest(pose_dict.keys(), timestamp)
        #         print("pose timestamp: \n", time_key)
        #         counter += 1
        #         # if time_key in image_dict:
        #         #     counter += 1
        #         #     print(counter)
        #         frame, up = write_image_and_pose(
        #             image_dict[timestamp], pose_dict[time_key], output_dir, counter, up)
        #         out["frames"].append(frame)

    plot_trajectory()
    print("done parsing the bag")
    nframes = len(out["frames"])
    up = up / np.linalg.norm(up)
    print("up vector was", up)
    # R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
    # R = np.pad(R, [0, 1])
    # R[-1, -1] = 1

    # for f in out["frames"]:
    #     f["transform_matrix"] = np.matmul(R, f["transform_matrix"])  # rotate up to be the z axis

    # # find a central point they are all looking at
    compute_center = False
    if compute_center == True:
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in out["frames"]:
            mf = f["transform_matrix"][0:3, :]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3, :]
                p, w = closest_point_2_lines(
                    mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
                if w > 0.00001:
                    totp += p*w
                    totw += w

        if totw > 0.0:
            totp /= totw
        print(totp)  # the cameras are looking at totp
        for f in out["frames"]:
            f["transform_matrix"][0:3, 3] -= totp

        avglen = 0.
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    for f in out["frames"]:
        if not isinstance(f["transform_matrix"], list):
            f["transform_matrix"] = f["transform_matrix"].tolist()

    #print(nframes, "frames")
    print(f"writing {OUT_PATH}")
    print("Saving JSON")
    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


if __name__ == "__main__":
    # Get the ROS bag file and output directory from the command line
    bag_file = "/media/saimouli/RPNG_FLASH_4/datasets/ar_table/bags/table_01.bag" #/media/saimouli/RPNG_FLASH_4/datasets/ar_table/bags/table_01.bag"
    output_dir_img = "/media/saimouli/RPNG_FLASH_4/datasets/ar_table/table_01_openvins_gt/rgb/"
    gt_file = "/home/saimouli/Desktop/catkin_ws/src/ov_nerf/ov_data/rpng_table/table_01.txt"
    start_time = 5.0
    end_time = 64.0
    AABB_SCALE = 16
    OUT_PATH = "/media/saimouli/RPNG_FLASH_4/datasets/ar_table/table_01_openvins_gt/transforms.json"
    kf_time = 1 # choose frames every x seconds

    # Camera calibration parameters
    # from camera_info topic
    angle_x = math.pi / 2
    w = 848.0
    h = 480.0
    fl_x = 418.97613525390625 #416.85223429743274
    fl_y = 418.5734558105469 #414.92069080087543
    k1 = -0.05735890567302704
    k2 = 0.06920064985752106
    k3 = -0.021814072504639626
    k4 = 0
    p1 = -0.0008085473091341555
    p2 = 0.0006239570793695748
    cx = 420.6703796386719
    cy = 237.04190063476562
    is_fisheye = False

    # fl = 0.5 * w / tan(0.5 * angle_x);
    angle_x = math.atan(w / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2
    fovx = angle_x * 180 / math.pi
    fovy = angle_y * 180 / math.pi

    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    out = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "k4": k4,
        "p1": p1,
        "p2": p2,
        "is_fisheye": is_fisheye,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "aabb_scale": AABB_SCALE,
        "frames": [],
    }

# Call the read_bag function
pose_topic = "/ov_nerf/poseimu"

read_bag(bag_file, output_dir_img, start_time, end_time, out, OUT_PATH, kf_time, pose_topic, interpolate_linear_poses = False, interpolate_cubic_poses_flag = False, gt_file = gt_file)
