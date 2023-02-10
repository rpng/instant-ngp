import cv2
import numpy as np

# Load the two images
img1 = cv2.imread('/media/rpng/RPNG_FLASH_4/datasets/nerf_dataset/fern/images_8/image001.png')
img2 = cv2.imread('/media/rpng/RPNG_FLASH_4/datasets/nerf_dataset/fern/images_8/image001.png')

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect keypoints and compute descriptors using ORB
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Match the descriptors using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort the matches in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

# Get the first N best matches
N = 200
best_matches = matches[:N]

# Extract the keypoints and use them to estimate the essential matrix
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

# Get the intrinsic camera matrix (assuming a square pixel)
camera_matrix = np.array([[412.4775270681586, 0, 258.2233135017183], [0, 418.54794926441633, 188.13696819178764], [0, 0, 1]])

# E, mask = cv2.findEssentialMat(src_pts, dst_pts, camera_matrix, cv2.RANSAC, 0.999, 1.0)

# # Use the essential matrix to get the relative rotation and translation
# _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, camera_matrix)

# # Print the relative rotation and translation
# print("Relative Rotation: \n", R)
# print("Relative Translation: \n", t)

# Draw the matches between the two images
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # Show the result
# cv2.imshow("Matches", img_matches)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Use PnP RANSAC to estimate the relative pose
dst_pts = dst_pts.astype('float64')
src_pts = src_pts.astype('float64')
camera_matrix = camera_matrix.astype('float64')
ret, rvec, tvec = cv2.solvePnPRansac(src_pts, dst_pts, camera_matrix, np.zeros(4).astype('float64'), flags=cv2.SOLVEPNP_ITERATIVE)

# Convert the rotation vector to rotation matrix
R, _ = cv2.Rodrigues(rvec)

# Print the relative rotation and translation
print("Relative Rotation: \n", R)
print("Relative Translation: \n", tvec)







