#!/usr/bin/env bash

data_dir="/home/wl/Desktop/rpng_plane"
gt_dir="/home/wl/workspace/ov_nerf/catkin_ws/src/ov_nerf/ov_data/rpng_table"
# bag
bagnames=(
"table_01"
"table_02"
"table_03"
"table_04"
"table_05"
"table_06"
"table_07"
"table_08"
)

# Loop through each rosbag and create NeRF weights!
for i in "${!bagnames[@]}"; do
  # Extract image files and match with ground truth pose
  # python3 openvins2nerf.py PATH_TO_GROUNDTRUTH PATH_TO_ROSBAG IMAGE_TOPIC INTRINSIC_TOPIC, GROUNDTRUTH_TOPIC
  rosbag_path=${data_dir}"/"${bagnames[i]}".bag"
  gt_path=${gt_dir}"/"${bagnames[i]}".txt"
  echo "Extracting images from ${bagnames[i]} and generating Jetson file."
  python3 openvins2nerf.py ${gt_path} ${rosbag_path} /d455/color/image_raw /d455/color/camera_info /ov_nerf/imupose &> /dev/null

  /home/wl/workspace/ov_nerf/catkin_ws/src/ov_nerf/thirdparty/instant_ngp/instant-ngp ${data_dir}"/"${bagnames[i]}"/transforms_vicon.json"
done
