
import json
import argparse, argcomplete
import numpy as np
import os
from pypcd4 import pypcd4
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import yaml

def cvat_to_openpcdet(pcd_path, json_path, params_path, dataset_root):
    if not (os.path.isdir(dataset_root)):
        os.makedirs(dataset_root)
    # Convert PCD files to BIN files
    ## Find all pcd files
    pcd_files = []
    for (path, dir, files) in os.walk(pcd_path):
        for filename in files:
            # print(filename)
            ext = os.path.splitext(filename)[-1]
            if ext == '.pcd':
                pcd_files.append(path + "/" + filename)

    ## Sort pcd files by file name
    pcd_files.sort()   

    ## Make bin_path directory
    if not (os.path.isdir(os.path.join(dataset_root, "velodyne"))):
        os.makedirs(os.path.join(dataset_root, "velodyne"))
    
    # Load the params file
    with open(params_path, 'r') as file:
        params_data = yaml.safe_load(file)
    rot = R.from_euler("xyz", params_data["rotation"], degrees=True)

    ## Converting Process
    for pcd_file in tqdm(pcd_files, desc="Converting PCD files to BIN"):
        ## Get pcd file
        pc = pypcd4.PointCloud.from_path(pcd_file)

        ## Generate bin file name
        pcd_file_number, _ = os.path.splitext(os.path.basename(pcd_file))
        bin_file_name = "{:06d}.bin".format(int(pcd_file_number))
        bin_file_path = os.path.join(dataset_root, "velodyne", bin_file_name)
        
        ## Get data from pcd
        np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
        np_i = (np.ones(len(pc.pc_data['z']), dtype=np.float32).astype(np.float32))

        xyz = np.transpose(np.vstack((np_x, np_y, np_z)))
        transformed_xyz = rot.apply(xyz)

        ## Stack all data    
        final_points = np.hstack((transformed_xyz, np.expand_dims(np_i, axis=1)), dtype=np.float32)

        ## Save bin file            
        final_points.tofile(bin_file_path)

    # Create annotations
    try:
        with open(json_path, 'r') as file:
            frames = json.load(file)["items"]
    except FileNotFoundError:
        print(f"Error: File '{json_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    if not (os.path.isdir(os.path.join(dataset_root, "label_2"))):
        os.makedirs(os.path.join(dataset_root, "label_2"))
    for frame in frames:
        frame_name = frame["id"]
        annotation_string = ""
        for annotation in frame["annotations"]:
            object_name = "Target" if annotation["label_id"] else "Pedestrian"
            object_center = annotation["position"]
            object_dims = annotation["scale"]
            # Apply rotation to bounding box
            yaw =  annotation["rotation"][2]
            yaw_quaternion = R.from_euler("z", yaw)
            rotation_quaternion = rot * yaw_quaternion
            # Compute the corners of the bounding box relative to the center
            length, width, height = object_dims
            box_corners = object_center + np.array([
                [ length / 2,  width / 2,  height / 2],
                [ length / 2,  width / 2, -height / 2],
                [ length / 2, -width / 2,  height / 2],
                [ length / 2, -width / 2, -height / 2],
                [-length / 2,  width / 2,  height / 2],
                [-length / 2,  width / 2, -height / 2],
                [-length / 2, -width / 2,  height / 2],
                [-length / 2, -width / 2, -height / 2]
            ])
            rotated_corners = rotation_quaternion.apply(box_corners)
            # Convert corners back to original representation
            rotated_center = np.mean(rotated_corners, axis=0)
            min_coords = np.min(rotated_corners, axis=0)
            max_coords = np.max(rotated_corners, axis=0)
            new_width = max_coords[0] - min_coords[0]  # Extent along x-axis
            new_length = max_coords[1] - min_coords[1]   # Extent along y-axis
            new_height = max_coords[2] - min_coords[2]  # Extent along z-axis
            # Apply rotation to bounding box
            rotated_center = rot.apply(object_center)
            annotation_string += "%s 0.0 0 0 0 0 50 50 %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n" % \
                                (object_name, new_height, new_width, new_length, 
                                 -rotated_center[1], -rotated_center[2], rotated_center[0], -yaw)
        with open(os.path.join(dataset_root, "label_2", "{:06d}.txt".format(int(frame_name))), "w") as f:
            f.write(annotation_string)
    
    # Create Images and Calibrations
    if not (os.path.isdir(os.path.join(dataset_root, "image_2"))):
        os.makedirs(os.path.join(dataset_root, "image_2"))
    if not (os.path.isdir(os.path.join(dataset_root, "calib"))):
        os.makedirs(os.path.join(dataset_root, "calib"))
    
    for pcd_file in pcd_files:
        pcd_file_number, _ = os.path.splitext(os.path.basename(pcd_file))
        image_file_name = "{:06d}.png".format(int(pcd_file_number))
        image_file_path = os.path.join(dataset_root, "image_2", image_file_name)
        black_image = Image.new("RGB", (1242, 375), color=(0,0,0))
        black_image.save(image_file_path)
        
        calib_file_name = "{:06d}.txt".format(int(pcd_file_number))
        calib_file_path = os.path.join(dataset_root, "calib", calib_file_name)

        with open(calib_file_path, 'w') as f:
            calib = dict({
                "P0" : np.eye(3,4).reshape((12,)),
                "P1" : np.eye(3,4).reshape((12,)),
                "P2" : np.eye(3,4).reshape((12,)),
                "P3" : np.eye(3,4).reshape((12,)),
                "R0_rect" : np.eye(3).reshape((9,)),
                "Tr_velo_to_cam" : np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                "Tr_imu_to_velo" : np.eye(3,4).reshape((12,))
            })
            for key, value in calib.items():
                f.write("%s: %s\n" % (key, " ".join(map(str, value))))





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert annotations from CVAT to the format needed for OpenPCDet")
    parser.add_argument("pcd_path", type=str, help="Path to directory containing PCD files")
    parser.add_argument("json_path", type=str, help="Path to Json file containing annotations from CVAT")
    parser.add_argument("params_path", type=str, help="Path to params file for the LiDAR sensor")
    parser.add_argument("dataset_root", type=str, help="Root directory to write dataset to")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    cvat_to_openpcdet(argdict["pcd_path"], argdict["json_path"], argdict["params_path"], argdict["dataset_root"])