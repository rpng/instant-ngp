import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def clean_noise(image):
    # Apply median blur to the image
    cleaned = cv2.medianBlur(image, 5)

    return cleaned

def sharpen_image(image):
    # Convert image to float
    image = np.float32(image)

    # Create a kernel to sharpen the image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Convolve the image with the kernel
    sharpened = cv2.filter2D(image, -1, kernel)

    # Convert back to 8-bit unsigned int
    sharpened = np.uint8(sharpened)

    return sharpened

# Load the two images
img1 = cv2.imread('/media/saimouli/RPNG_FLASH_4/datasets/nerf_dataset/room/images_4/DJI_20200226_143850_006.png')
#img2_render = cv2.imread('/media/saimouli/RPNG_FLASH_4/datasets/nerf_dataset/room/images_4/screenshot/DJI_20200226_143855_969.png')
img2 = cv2.imread('/media/saimouli/RPNG_FLASH_4/datasets/nerf_dataset/room/images_4/DJI_20200226_143855_969.png')

#img2 = img2_render

img1 = clean_noise(img1)
img2 = clean_noise(img2)

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

gray1 = sharpen_image(gray1)
gray2 = sharpen_image(gray2)

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
N = 60
best_matches = matches[:N]

# Draw the matches between the two images
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show the result
cv2.imshow("Matches", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract the keypoints and use them to estimate the essential matrix
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

# Get the intrinsic camera matrix (assuming a square pixel)
fx = 399.29145295196014
fy = 394.82594561727626
cx = 275.4910414710184
cy = 178.6618379675057
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

E, mask = cv2.findEssentialMat(src_pts, dst_pts, camera_matrix, cv2.RANSAC, 0.999, 1.0)

# Use the essential matrix to get the relative rotation and translation
_, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, camera_matrix)

# Print the relative rotation and translation
print("Relative Rotation: \n", R)
print("Relative Translation: \n", t)




