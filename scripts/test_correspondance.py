import cv2
import numpy as np

# Load the two images
img1 = cv2.imread('/media/rpng/RPNG_FLASH_4/datasets/nerf_dataset/fern/images_4/screenshot/image001.png')
img2 = cv2.imread('/media/rpng/RPNG_FLASH_4/datasets/nerf_dataset/fern/images_4/image001.png')

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

# Draw the matches between the two images
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show the result
cv2.imshow("Matches", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()







