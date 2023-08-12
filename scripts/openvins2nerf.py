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
import shutil

cam2wld_traj = []
wld2cam_traj = []

class Options:
    def __init__(self, args):
        # Sanity check
        if len(args) < 8:
            print ('You need to pass the rosbag file path: python3 openvins2nerf.py PATH_TO_GROUNDTRUTH PATH_TO_ROSBAG IMAGE_TOPIC INTRINSIC_TOPIC, GROUNDTRUTH_TOPIC')
            exit(0)

        self.intrinsic = []
        self.groundtruth_path = args[1]
        self.rosbag_path = args[2]
        self.image_topic = args[3]
        self.intrinsic_topic = args[4]
        self.T_ItoC = np.array(list(np.float_(args[5].split(",")))).reshape(4,4)  # T_CtoI ? T_ItoC ?
        self.json_path = args[6]
        self.srp_thr = float(args[7])
        self.ori_thr = float(args[8])
        self.pos_thr = float(args[9])
        rosbag_name = os.path.splitext(os.path.basename(self.rosbag_path))[0]
        rosbag_dir = os.path.dirname(self.rosbag_path)
        self.out_img_dir = rosbag_dir + "/" + rosbag_name + "/images_s" + args[7] + "_o" + args[8] + "_p" + args[9]

        # Sanity check of the files
        if not os.path.exists(self.groundtruth_path):
            print ("The file does not exist: ", self.groundtruth_path)
            exit(0)

        if not os.path.exists(self.rosbag_path):
            print ("The file does not exist: ", self.rosbag_path)
            exit(0)

        if os.path.splitext(os.path.basename(self.groundtruth_path))[1] != ".txt":
            print ("This is not ground truth file: ", self.rosbag_path)
            exit(0)

        if os.path.splitext(os.path.basename(self.rosbag_path))[1] != ".bag":
            print ("This is not rosbag file: ", self.rosbag_path)
            exit(0)

        # Create the output image path if not exist
        if os.path.exists(self.out_img_dir):
            shutil.rmtree(self.out_img_dir)
        os.makedirs(self.out_img_dir)

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

def log_so3(R):
    # note switch to base 1
    R11 = R[0, 0]
    R12 = R[0, 1]
    R13 = R[0, 2]
    R21 = R[1, 0]
    R22 = R[1, 1]
    R23 = R[1, 2]
    R31 = R[2, 0]
    R32 = R[2, 1]
    R33 = R[2, 2]

    # Get trace(R)
    tr = R.trace();
    omega = [];

    # when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, etc.
    # we do something special
    if tr + 1.0 < 1e-10:
        if np.abs(R33 + 1.0) > 1e-5:
            omega = (np.pi / np.sqrt(2.0 + 2.0 * R33)) * np.array([R13, R23, 1.0 + R33]).reshape(3,1)
        elif np.abs(R22 + 1.0) > 1e-5:
           omega = (np.pi / np.sqrt(2.0 + 2.0 * R22)) * np.array([R12, 1.0 + R22, R32]).reshape(3,1)
        else:
            omega = (np.pi / np.sqrt(2.0 + 2.0 * R11)) * np.array([1.0 + R11, R21, R31]).reshape(3,1)
    else:
        magnitude = []
    tr_3 = tr - 3.0; # always negative
    if tr_3 < -1e-7:
        if tr < -1:
            tr = -1
        theta = np.arccos((tr - 1.0) / 2.0);
        magnitude = theta / (2.0 * np.sin(theta))
    else:
        magnitude = 0.5 - tr_3 / 12.0
    omega = magnitude * np.array([R32 - R23, R13 - R31, R21 - R12]).reshape(3,1)

    return omega;

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

def compute_poses_to_plot(options, pose):
    R_ItoG = pose[:3, :3]
    p_IinG = pose[:3, 3].reshape([3, 1]) #in reality it is p_IinG
    
    p_IinC = options.T_ItoC[:3, 3] #in reality it is p_IinC
    R_ItoC = options.T_ItoC[:3, :3] #in reality it is R_ItoC

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
def write_image_and_pose(image, pose, options, counter, up):
    # Get the timestamp from the pose message
    # timestamp = pose.header.stamp.to_sec()
    # Convert the image message to a cv2 image
    image = CvBridge().imgmsg_to_cv2(image, "bgr8")
    # Construct the file names
    image_file = os.path.join(options.out_img_dir, f"frame_{counter}.png")

    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    
    sharp_img = sharpness(image)
    R_ItoG = np.array(pose[:3, :3])
    p_IinG = pose[:3, 3].reshape([3, 1]) #in reality it is p_IinG

    p_IinC = options.T_ItoC[:3, 3].reshape([3,1]) #in reality it is p_IinC
    R_ItoC = options.T_ItoC[:3, :3] #in reality it is R_ItoC

    p_GinI = -R_ItoG.transpose() @ p_IinG
    R_CtoI = R_ItoC.transpose()

    # transform for cam2wld
    p_GinC = R_ItoC @ p_GinI + p_IinC.reshape([3,1]) #wld2cam
    R_CtoG = R_ItoG @ R_CtoI
    
    p_CinG = -R_CtoG @ p_GinC

    #R_GtoN = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
    #R_CtoN = R_GtoN @ R_CtoG
    c2w = np.concatenate([np.concatenate([R_CtoG, p_CinG], 1), bottom], 0) # vicon frame is world frame


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

def read_bag(options):
    # Read the ground truth file
    pose_gt_dict = {}
    timestamps_gt = []; poses_gt = []
    totoal_cnt = 0
    sharp_rej = 0;
    pose_rej = 0;
    pass_cnt = 0;
    with open(options.groundtruth_path, "r") as f:
        next(f)
        for line in f:
            line = line.strip()
            data = line.split(" ")
            timestamps_gt.append(float(data[0]))
            poses_gt.append([float(x) for x in data[1:]])
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
    timestamps_gt = np.array(timestamps_gt)
    poses_gt = np.array(poses_gt)
    # print("Finish reading ground truth.")


    first_pose = True
    frames_counter = 0
    up = np.zeros(3)
    # print("Reading bag, hold tight!")
    time_temp = 0
    poses_used = [];
    # Open the ROS bag
    with rosbag.Bag(options.rosbag_path, "r") as bag:

                # Get the poses
        pose_dict = pose_gt_dict
        
        # Read all the imaes
        image_dict = {}
        bridge = CvBridge() # Create a bridge for converting image messages
        for topic, msg, t in bag.read_messages():
            if topic == options.image_topic:
                totoal_cnt += 1
                # skip if image is very blurry
                if sharpness(bridge.imgmsg_to_cv2(msg, "bgr8")) < options.srp_thr:
                    sharp_rej += 1
                    continue;
                image_dict[msg.header.stamp.to_sec()] = msg

        # print("# of poses: ", len(pose_dict))
        # print("# of imges: ", len(image_dict))


        poses_used = [] # Poses used
        #find the nearest timestamp of pose based on images
        pose_keys_sorted = sorted(pose_dict.keys())
        for timestamp in sorted(image_dict.keys()):
            # Get current time and pose
            curr_pose_time = nearest(pose_dict.keys(), timestamp)
            curr_pose = ros_to_pose_matrix(pose_dict[curr_pose_time])
            # print("Img timestamp: {} Pos timestamp: {} difference: {} ms".format(timestamp, curr_pose_time, abs(curr_pose_time - timestamp) * 1000))

            # skip if we have close pose
            if poses_used:
                R_ItoG_curr = np.array(curr_pose[:3, :3])
                p_IinG_curr = curr_pose[:3, 3].reshape([3, 1])
                found_close_pose = False
                for pose in poses_used:
                    R_ItoG = np.array(pose[:3, :3])
                    p_IinG = pose[:3, 3].reshape([3, 1]) #in reality it is p_IinG
                    o_diff =   np.linalg.norm(log_so3(R_ItoG_curr @ R_ItoG.transpose()),2)
                    p_diff = np.linalg.norm((p_IinG_curr - p_IinG),2)
                    if o_diff < options.ori_thr and p_diff < options.pos_thr:
                        # print("o diff: {} < {}, p diff: {} < {}".format(o_diff, options.ori_thr, p_diff, options.pos_thr))
                        found_close_pose = True
                        break
                if found_close_pose:
                    pose_rej += 1
                    continue

            frame, up = write_image_and_pose(image_dict[timestamp], curr_pose, options, len(poses_used), up)
            compute_poses_to_plot(options, curr_pose)
            options.intrinsic["frames"].append(frame)
            poses_used.append(curr_pose)
            pass_cnt += 1

    # plot_trajectory()
    print("done parsing the bag. totoal {}, sharp rej {}, pose rej {}, pass {}".format(totoal_cnt, sharp_rej, pose_rej, pass_cnt))
    nframes = len(options.intrinsic["frames"])
    up = up / np.linalg.norm(up)
    # print("up vector was", up)
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
        for f in options.intrinsic["frames"]:
            mf = f["transform_matrix"][0:3, :]
            for g in options.intrinsic["frames"]:
                mg = g["transform_matrix"][0:3, :]
                p, w = closest_point_2_lines(
                    mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
                if w > 0.00001:
                    totp += p*w
                    totw += w

        if totw > 0.0:
            totp /= totw
        print(totp)  # the cameras are looking at totp
        for f in options.intrinsic["frames"]:
            f["transform_matrix"][0:3, 3] -= totp

        avglen = 0.
        for f in options.intrinsic["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        for f in options.intrinsic["frames"]:
            f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    for f in options.intrinsic["frames"]:
        if not isinstance(f["transform_matrix"], list):
            f["transform_matrix"] = f["transform_matrix"].tolist()

    #print(nframes, "frames")
    # print(f"writing {options.json_path}")
    # print("Saving JSON")
    with open(options.json_path, "w") as outfile:
        json.dump(options.intrinsic, outfile, indent=2)


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

def get_cam_intrinsic(rosbag_path, intrinsic_topic):
    bag = rosbag.Bag(rosbag_path)
    intrinsic = ""
    found = False
    AABB_SCALE = 128
    for topic, msg, t in bag.read_messages(intrinsic_topic):
        w = msg.width
        h = msg.height
        fl_x = msg.K[0]
        fl_y = msg.K[4]
        k1 = msg.D[0]
        k2 = msg.D[1]
        k3 = msg.D[4] # ??
        k4 = 0
        p1 = msg.D[2]
        p2 = msg.D[3]
        cx = msg.K[2]
        cy = msg.K[5]
        is_fisheye = False # if plumb_bob: false else true?

        # fl = 0.5 * w / tan(0.5 * angle_x);
        angle_x = math.atan(w / (fl_x * 2)) * 2
        angle_y = math.atan(h / (fl_y * 2)) * 2

        intrinsic = {
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
        found = True
        break
    bag.close()
    if found:
        return intrinsic
    else:
        # This is euroc
        intrinsic = {
            "camera_angle_x": 1.3631246603369132,
            "camera_angle_y": 0.9593372178212194,
            "fl_x": 463.4830598553089,
            "fl_y": 461.3701440000305,
            "k1": -0.27191566105980836,
            "k2": 0.06426776880177594,
            "k3": 0,
            "k4": 0,
            "p1": 0.0016697300239062935,
            "p2": -0.0002050480367216984,
            "is_fisheye": False,
            "cx": 372.1892110447352,
            "cy": 240.22233824498292,
            "w": 752,
            "h": 480,
            "aabb_scale": AABB_SCALE,
            "frames": [],
        }
        return intrinsic



if __name__ == "__main__":
    # Create a class that contains all the options!
    options = Options(sys.argv)

    # Get camera intrinsic in json format
    options.intrinsic = get_cam_intrinsic(options.rosbag_path, options.intrinsic_topic)

    read_bag(options)
