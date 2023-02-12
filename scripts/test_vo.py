import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Read images from directory
def read_image_path(path):
    image_paths = []

    # Loop through all the files in the directory
    for filename in os.listdir(path):
        # Check if the file is an image
        if filename.endswith(".png"):
            # Construct the full path of the image
            full_path = os.path.join(path, filename)
            # Append the full path to the list
            image_paths.append(full_path)
    return image_paths


def perform_VO(image_path):
    # Camera intrinsic parameters
    K = np.array([[412.4775270681586, 0, 258.2233135017183], [0, 418.54794926441633, 188.13696819178764], [0, 0, 1]])

    # Estimate camera poses using the essential matrix
    previous_frame = cv2.imread(images_path[0])
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    R_list = []
    t_list = []
    for path in images_path[1:]:
        # Convert to grayscale
        current_frame = cv2.imread(path)
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        if previous_frame is not None:
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
                # plt.imshow(img_match)
                #plt.show()
                
                
        previous_frame = current_frame_gray

    # Plot the trajectory of the camera motion
    cumulative_R = np.eye(3)
    cumulative_t = np.zeros((3, 1))
    trajectory = np.zeros((3, len(R_list) + 1))
    trajectory[:, 0] = cumulative_t[:, 0]
    for i in range(len(R_list)):
        cumulative_R = np.matmul(cumulative_R, R_list[i])
        cumulative_t = cumulative_t + np.matmul(cumulative_R, t_list[i])
        trajectory[:, i + 1] = cumulative_t[:, 0]

    return trajectory


images_path = read_image_path("/media/saimouli/RPNG_FLASH_4/datasets/nerf_dataset/fern/images_8/")
images_path_render = read_image_path("/media/saimouli/RPNG_FLASH_4/datasets/nerf_dataset/fern/images_8/screenshot")
trajectory = perform_VO(images_path)
trajectory_render = perform_VO(images_path_render)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
render_vo, = ax.plot(trajectory_render[0, :], trajectory_render[1, :], trajectory_render[2, :], 'ro', label='VO render imgs')
img_vo, = ax.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :], 'b+', label='VO original imgs')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(['VO render imgs', 'VO original imgs'])
plt.show()
