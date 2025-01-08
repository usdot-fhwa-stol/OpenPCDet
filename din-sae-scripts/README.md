# DIN SAE Scripts

This folder contains scripts used to evaluate the DIN SAE Automotive LiDAR specification on KITTI-formatted LiDAR data.
These include:
- `cvat_to_openpcdet.py`: Used to convert annotations from the Datamuro format exported by [CVAT](https://www.cvat.ai/post/3d-point-cloud-annotation) to the KITTI 3D Object dataset format. This includes:
  1. `velodyne/`: LiDAR data in `.bin` format
  2. `label_2/`: Bounding box information
  3. `image_2/`: A camera image of the scene (since there are no images in the provided dataset, these are all black)
  4. `calib/`: Transformation matrices for going from LiDAR to camera coordinates

  Usage: `python3 cvat_to_openpcdet.py {pcd_path} {json_path} {params_path} {dataset_root}`
    1. `pcd_path`: Path to directory containing PCD files
    2. `json_path`: Path to Json file containing annotations from CVAT
    3. `params_path`: Path to params file for the LiDAR sensor detailing transformation needed
    4. `dataset_root`: Root directory to write KITTI-formatted dataset to

Installation instructions can be found [here](../docs/HASS_SUPPORT.md).