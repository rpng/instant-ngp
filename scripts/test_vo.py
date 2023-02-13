import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import quaternion

# Read images from directory
def read_image_path(path):
    image_paths = []

    # Loop through all the files in the directory
    for filename in sorted(os.listdir(path)):
        # Check if the file is an image
        if filename.endswith(".png"):
            # Construct the full path of the image
            full_path = os.path.join(path, filename)
            # Append the full path to the list
            image_paths.append(full_path)
    return image_paths

def visualize_img_seq(image_path):
    for path in image_path:
        img = cv2.imread(path)
        cv2.imshow("vis", img)
        cv2.waitKey(1)

def parse_gt_traj(gt_path):
    poses = []
    with open(gt_path, 'r') as f:
        for i in range(3):
            f.readline()
        for line in f:
            values = line.strip().split()
            poses.append([float(v) for v in values[1:]])

    poses = np.array(poses)
    return poses

def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1) - rot * model.mean(1)

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(
        np.sum(np.multiply(alignment_error, alignment_error), 0)
    ).A[0]

    return rot, trans, trans_error

def perform_VO(image_path):
    # Camera intrinsic parameters
    K = np.array([[412.4775270681586, 0, 258.2233135017183], [0, 418.54794926441633, 188.13696819178764], [0, 0, 1]])

    # Estimate camera poses using the essential matrix
    # previous_frame = cv2.imread(images_path[0])
    # previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = None
    R_list = []
    t_list = []
    for path in image_path:
        # Convert to grayscale
        current_frame = cv2.imread(path)
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        if previous_frame_gray is not None:
            # Find keypoints and descriptors
            sift = cv2.xfeatures2d.SIFT_create()
            kp1, des1 = sift.detectAndCompute(previous_frame_gray, None)
            kp2, des2 = sift.detectAndCompute(current_frame_gray, None)
            
            # Match keypoints
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Select good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
                    
            if len(good_matches) >= 8:
                # Extract matching keypoints
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                
                # Compute essential matrix
                E, mask = cv2.findEssentialMat(pts1, pts2, K)
                
                # Decompose essential matrix to get rotation and translation
                _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
                
                # Save rotation and translation
                R_list.append(R)
                t_list.append(t)

                # Plot the matches between previous frame and current frame
                img_match = cv2.drawMatches(previous_frame_gray, kp1, current_frame_gray, kp2, good_matches, None, flags=2)
                cv2.imshow("mathces", img_match)
                cv2.waitKey(1)
                #plt.show()       
        previous_frame_gray = current_frame_gray

    # Plot the trajectory of the camera motion
    cumulative_R = np.eye(3)
    cumulative_t = np.zeros((3, 1))
    trajectory = np.zeros((3, len(R_list) + 1))
    trajectory[:, 0] = cumulative_t[:, 0]
    filename = "/media/saimouli/RPNG_FLASH_4/datasets/rgbd_dataset_freiburg3_long_office_household/poses_vo.txt"
    with open(filename, 'w') as f:
        for i in range(len(R_list)):
            cumulative_R = np.matmul(cumulative_R, R_list[i])
            cumulative_t = cumulative_t + np.matmul(cumulative_R, t_list[i])
            trajectory[:, i + 1] = cumulative_t[:, 0]
            q = quaternion.from_rotation_matrix(cumulative_R)
            f.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(cumulative_t[0,0], cumulative_t[1,0], cumulative_t[2,0], q.x, q.y, q.z, q.w))
    return trajectory


images_path = read_image_path("/media/saimouli/RPNG_FLASH_4/datasets/rgbd_dataset_freiburg3_long_office_household/rgb/")
#visualize_img_seq(images_path)
#images_path_render = read_image_path("/media/saimouli/RPNG_FLASH_4/datasets/nerf_dataset/fern/images_8/screenshot")
trajectory = perform_VO(images_path)
#trajectory_render = perform_VO(images_path_render)

gt_poses = parse_gt_traj("/media/saimouli/RPNG_FLASH_4/datasets/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt")
fig, ax = plt.subplots()
gt_pose, = ax.plot(gt_poses[:, 0], gt_poses[:, 1], 'r+', label='GT')
# img_vo, = ax.plot(trajectory[0, :], trajectory[1, :], 'b', label='VO original imgs')
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax.legend(['VO GT', 'VO original imgs'])

plt.figure(2)
plt.plot(trajectory[0, :], trajectory[1, :], 'b')
plt.show()