#!/usr/bin/env bash
clear
# build python
(cd /home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/thirdparty/instant_ngp && cmake --build build --config RelWithDebInfo -j)
# build catkin
catkin build

### TRAIN CONFIGURATION
# bag
train_bag_path=(
"/home/rpng/datasets/rpng_plane/table_01.bag"
"/home/rpng/datasets/euroc_mav/V1_01_easy.bag"
)
# gt
train_gt_path=(
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/ov_data/rpng_table/table_01.txt"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/ov_data/euroc_mav/V1_01_easy.txt"
)
# image topic
train_img_topic=(
"/d455/color/image_raw"
"/cam0/image_raw"
)
# intrinsic topic
train_int_topic=(
"/d455/color/camera_info"
"/d455/color/camera_info" # THis is fake. euroc mav does not have topic for camera intrinsic
)
# output dir
train_output_dir=(
"/home/rpng/datasets/rpng_plane/table_01"
"/home/rpng/datasets/euroc_mav/V1_01_easy"
)
# Extrinsic
train_ext=(
"0.9999654398038452,0.007342326779113337,-0.003899927610975742,-0.027534314618518095,-0.0073452195116216765,0.9999727585590525,-0.0007279355223411334,-0.0030587146933711722,0.0038944766308488753,0.0007565561891287445,0.9999921303062861,-0.023605118842939803,0,0,0,1"
"0.0148655429818,-0.999880929698,0.00414029679422,-0.0216401454975,0.999557249008,0.0149672133247,0.025715529948,-0.064676986768,-0.0257744366974,0.00375618835797,0.999660727178,0.00981073058949,0.0,0.0,0.0,1.0"
)

# image sharpness
train_sharpness=(
"200"
"400"
#"1000"
#"2000"
)

# ori threshold
train_ori=(
"1"
"0.5"
"0.1"
#"0.05"
#"0.01"
)

train_pos=(
"1"
"0.5"
#"0.3"
"0.1"
)


### TEST CONFIGURATION
# test name
test_dataset=(
"table_01"
"table_02"
"table_03"
"table_04"
"table_05"
"table_06"
"table_07"
"table_08"
"V1_01_easy"
"V1_02_medium"
"V1_03_difficult"
)

# bag
test_bag_path=(
"/home/rpng/datasets/rpng_plane/table_01.bag"
"/home/rpng/datasets/rpng_plane/table_02.bag"
"/home/rpng/datasets/rpng_plane/table_03.bag"
"/home/rpng/datasets/rpng_plane/table_04.bag"
"/home/rpng/datasets/rpng_plane/table_05.bag"
"/home/rpng/datasets/rpng_plane/table_06.bag"
"/home/rpng/datasets/rpng_plane/table_07.bag"
"/home/rpng/datasets/rpng_plane/table_08.bag"
"/home/rpng/datasets/euroc_mav/V1_01_easy.bag"
"/home/rpng/datasets/euroc_mav/V1_02_medium.bag"
"/home/rpng/datasets/euroc_mav/V1_03_difficult.bag"
)

# gt
test_gt_path=(
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/ov_data/rpng_table/table_01.csv"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/ov_data/rpng_table/table_02.csv"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/ov_data/rpng_table/table_03.csv"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/ov_data/rpng_table/table_04.csv"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/ov_data/rpng_table/table_05.csv"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/ov_data/rpng_table/table_06.csv"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/ov_data/rpng_table/table_07.csv"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/ov_data/rpng_table/table_08.csv"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/ov_data/euroc_mav/V1_01_easy.csv"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/ov_data/euroc_mav/V1_02_medium.csv"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/ov_data/euroc_mav/V1_03_difficult.csv"
)

# gt
test_config_path=(
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/config/rpng_table/estimator_config.yaml"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/config/rpng_table/estimator_config.yaml"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/config/rpng_table/estimator_config.yaml"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/config/rpng_table/estimator_config.yaml"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/config/rpng_table/estimator_config.yaml"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/config/rpng_table/estimator_config.yaml"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/config/rpng_table/estimator_config.yaml"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/config/rpng_table/estimator_config.yaml"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/config/euroc_mav/estimator_config.yaml"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/config/euroc_mav/estimator_config.yaml"
"/home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/config/euroc_mav/estimator_config.yaml"
)

#estimation mode
nerf_depth_enabled=(
   "false"
   "true"
)
nerf_n_match=(
  "1"
  "3"
  "5"
)

#estimation mode
nerf_depth_enabled=(
   "false"
   "true"
)
nerf_n_match=(
  "1"
  "3"
  "5"
)


# Just Regular OpenVINS
#for i in {0..10}; do
#  start_time="$(date -u +%s)"
#  roslaunch ov_nerf serial.launch \
#  nerf_enabled:="false" \
#  multithread:="false" \
#  dataset:=${test_dataset[i]} \
#  path_bag:=${test_bag_path[i]} \
#  path_gt:=${test_gt_path[i]} \
#  config_path:=${test_config_path[i]} \
#  save_dir:="/home/rpng/Documents/woosik_ws/ov_nerf_results/OpenVINS"  &> /dev/null
#  echo "BASH: OpenVINS ${test_dataset[i]} took $(($(date -u +%s)-$start_time)) seconds";
#done

# Loop through each rosbag and create NeRF weights!
for i in {1..1}; do
  for j in "${!train_sharpness[@]}"; do
    for k in "${!train_ori[@]}"; do
      for l in "${!train_pos[@]}"; do
        # Extract image files and match with ground truth pose
        config="s${train_sharpness[j]}_o${train_ori[k]}_p${train_pos[l]}"
        json=${train_output_dir[i]}"/vicon_$config.json"

#        echo "Extracting images from ${train_bag_path[i]} with settings $config."
#        python3 /home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/thirdparty/instant_ngp/scripts/openvins2nerf.py \
#        ${train_gt_path[i]} ${train_bag_path[i]} ${train_img_topic[i]} ${train_int_topic[i]} ${train_ext[i]} ${json} ${train_sharpness[j]} ${train_ori[k]} ${train_pos[l]} #&> /dev/null

        # Perform training
#        echo "NeRF training ${json}."
#        /home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/thirdparty/instant_ngp/scripts/run.py \
#        ${json} --save_snapshot ${train_output_dir[i]}"/vicon_$config.ingp"

        # Test algtrain_orithms
        for m in {8..10}; do
          for n in "${!nerf_depth_enabled[@]}"; do
            for o in "${!nerf_n_match[@]}"; do
              # Skip in case mode is nonechange or downsample and this is not the first run ("${j}" != "00").
              # This is because there is no randomness in estimation thus no need to run time-consuming redundant exp.
              if [ "${nerf_depth_enabled[n]}" == "true" ]
                then
                  if [ "${nerf_n_match[o]}" == "3" ] || [ "${nerf_n_match[o]}" == "5" ]
                  then continue;
                fi;
              fi;
              start_time="$(date -u +%s)"
              roslaunch ov_nerf serial.launch \
              multithread:="false" \
              nerf_map_path:=${train_output_dir[i]}"/vicon_$config.ingp" \
              dataset:=${test_dataset[m]} \
              path_bag:=${test_bag_path[m]} \
              path_gt:=${test_gt_path[m]} \
              config_path:=${test_config_path[m]} \
              nerf_depth_enabled:=${nerf_depth_enabled[n]} \
              nerf_n_match:=${nerf_n_match[o]} \
              save_dir:="/home/rpng/Documents/woosik_ws/ov_nerf_results/vicon_${config}_d_${nerf_depth_enabled[n]}_n_${nerf_n_match[o]}"  &> /dev/null
              echo "BASH: OV_NERF ${test_dataset[m]} vicon_${config}_d_${nerf_depth_enabled[n]}_n_${nerf_n_match[o]} took $(($(date -u +%s)-$start_time)) seconds";
            done
          done
        done
      done
    done
  done
done


#echo "rosrun ov_eval error_comparison posyaw /home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/ov_data/euroc_table/ /home/rpng/Documents/woosik_ws/ov_nerf_results/"
#rosrun ov_eval error_comparison posyaw /home/rpng/Documents/woosik_ws/catkin_ws/src/ov_nerf/ov_data/euroc_table/ /home/rpng/Documents/woosik_ws/ov_nerf_results/
