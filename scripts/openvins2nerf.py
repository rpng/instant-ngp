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


def write_image_and_pose(image, pose, output_dir, counter, up):
    # Get the timestamp from the pose message
    # timestamp = pose.header.stamp.to_sec()
    # Convert the image message to a cv2 image
    image = CvBridge().imgmsg_to_cv2(image, "bgr8")
    # Construct the file names
    image_file = os.path.join(output_dir, f"frame_{counter}.png")

    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    
    b = sharpness(image)
    print(image_file, "sharpness=", b)
    tvec = np.array([pose.position.x, pose.position.y, pose.position.z])
    qvec = np.array([pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w])  # x,y,z,w

    R = quat2rot(qvec[0], qvec[1], qvec[2], qvec[3])
    t = tvec.reshape([3, 1])

    # rigid transformation from imu2cam

    m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
    c2w = np.linalg.inv(m)

    # convert to nerf coordinates
    c2w[0:3, 2] *= -1  # flip the y and z axis
    c2w[0:3, 1] *= -1
    c2w = c2w[[1, 0, 2, 3], :]
    c2w[2, :] *= -1  # flip whole world upside down
    up += c2w[0:3, 1]

    frame = {"file_path": image_file, "sharpness": b, "transform_matrix": c2w}

    cv2.imwrite(image_file, image)

    return frame, up
    #out["frames"].append(frame)

    
    # cv2.imwrite(image_file, image)


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


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

def read_bag(bag_file, output_dir, start_time, end_time, out, OUT_PATH):
    first_pose = True
    up = np.zeros(3)
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
            if start_time <= t.to_sec() - start_pose_time <= end_time:
                if topic == "/d455/color/image_raw":
                    # Store the image message in the dictionary
                    image_dict[msg.header.stamp.to_sec()] = msg
                elif topic == "/ov_msckf/poseimu":
                    # Store the pose message in the dictionary
                    pose_dict[msg.header.stamp.to_sec()] = msg.pose.pose

        print("pose len: ", len(pose_dict))
        print("img len: ", len(image_dict))
        # Iterate over the timestamps
        counter = 0
        
        for timestamp in sorted(pose_dict.keys()):
            print("pose timestamp: ", timestamp)
            time_key = nearest(image_dict.keys(), timestamp)
            print("img timestamp: \n", time_key)
            counter += 1
            # if time_key in image_dict:
            #     counter += 1
            #     print(counter)
            frame, up = write_image_and_pose(
                image_dict[time_key], pose_dict[timestamp], output_dir, counter, up)
            out["frames"].append(frame)

    print("done parsing the bag")
    nframes = len(out["frames"])
    up = up / np.linalg.norm(up)
    print("up vector was", up)
    R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(
            R, f["transform_matrix"])  # rotate up to be the z axis

    # find a central point they are all looking at
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
        f["transform_matrix"] = f["transform_matrix"].tolist()

    print(nframes, "frames")
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
    bag_file = "/media/saimouli/RPNG_FLASH_4/nerf_ws/src/table_01_gt.bag"
    output_dir_img = "/media/saimouli/RPNG_FLASH_4/datasets/ar_table/table_01_openvins/rgb/"
    start_time = 10.0
    end_time = 20.0
    AABB_SCALE = 16
    OUT_PATH = "/media/saimouli/RPNG_FLASH_4/datasets/ar_table/table_01_openvins/transforms.json"

    # Camera calibration parameters
    # from camera_info topic
    angle_x = math.pi / 2
    w = 848.0
    h = 480.0
    fl_x = 418.97613525390625
    fl_y = 418.5734558105469
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
read_bag(bag_file, output_dir_img, start_time, end_time, out, OUT_PATH)
