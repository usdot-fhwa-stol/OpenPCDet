
import json
import argparse, argcomplete
import numpy as np
import os
from pypcd4 import pypcd4
from PIL import Image
from tqdm import tqdm

def cvat_to_openpcdet(pcd_path, json_path):
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
    if not (os.path.isdir("velodyne")):
        os.makedirs("velodyne")

    ## Converting Process
    for pcd_file in tqdm(pcd_files, desc="Converting PCD files to BIN"):
        ## Get pcd file
        pc = pypcd4.PointCloud.from_path(pcd_file)

        ## Generate bin file name
        pcd_file_number, _ = os.path.splitext(os.path.basename(pcd_file))
        bin_file_name = "{:04d}.bin".format(int(pcd_file_number))
        bin_file_path = os.path.join("velodyne", bin_file_name)
        
        ## Get data from pcd (x, y, z, intensity, ring, time)
        np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
        np_i = (np.ones(len(pc.pc_data['z']), dtype=np.float32).astype(np.float32))
        #(np.array(pc.pc_data['intensity'], dtype=np.float32)).astype(np.float32)/256
        # np_r = (np.array(pc.pc_data['ring'], dtype=np.float32)).astype(np.float32)
        # np_t = (np.array(pc.pc_data['time'], dtype=np.float32)).astype(np.float32)

        ## Stack all data    
        points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))

        ## Save bin file            
        points_32.tofile(bin_file_path)

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
    if not (os.path.isdir("label_2")):
        os.makedirs("label_2")
    for frame in frames:
        frame_name = frame["id"]
        annotation_string = ""
        for annotation in frame["annotations"]:
            object_name = "Target" if annotation["label_id"] else "Pedestrian"
            object_center = annotation["position"]
            object_dims = annotation["scale"]
            yaw_angle = annotation["rotation"][2]
            annotation_string += "%s 0.0 0 0 0 0 50 50 %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n" % \
                                (object_name, object_dims[0], object_dims[1], object_dims[2], 
                                 object_center[0], object_center[1], object_center[2], yaw_angle)
        with open(os.path.join("label_2", "{:04d}.txt".format(int(frame_name))), "w") as f:
            f.write(annotation_string)
    
    # Create Images and Calibrations
    if not (os.path.isdir("image_2")):
        os.makedirs("image_2")
    if not (os.path.isdir("calib")):
        os.makedirs("calib")
    
    for pcd_file in pcd_files:
        pcd_file_number, _ = os.path.splitext(os.path.basename(pcd_file))
        image_file_name = "{:04d}.png".format(int(pcd_file_number))
        image_file_path = os.path.join("image_2", image_file_name)
        black_image = Image.new("RGB", (1242, 375), color=(0,0,0))
        black_image.save(image_file_path)
        
        calib_file_name = "{:04d}.txt".format(int(pcd_file_number))
        calib_file_path = os.path.join("calib", calib_file_name)

        with open(calib_file_path, 'w') as f:
            calib = dict({
                "P0" : np.eye(3,4).reshape((12,)),
                "P1" : np.eye(3,4).reshape((12,)),
                "P2" : np.eye(3,4).reshape((12,)),
                "P3" : np.eye(3,4).reshape((12,)),
                "R0_rect" : np.eye(3).reshape((9,)),
                "Tr_velo_to_cam" : np.array([6.927964000000e-03, -9.999722000000e-01, -2.757829000000e-03, -2.457729000000e-02, -1.162982000000e-03, 2.749836000000e-03, -9.999955000000e-01, -6.127237000000e-02, 9.999753000000e-01, 6.931141000000e-03, -1.143899000000e-03, -3.321029000000e-01]),
                "Tr_imu_to_velo" : np.eye(3,4).reshape((12,))
            })
            for key, value in calib.items():
                f.write("%s: %s\n" % (key, " ".join(map(str, value))))





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert annotations from CVAT to the format needed for OpenPCDet")
    parser.add_argument("pcd_path", type=str, help="Path to directory containing PCD files")
    parser.add_argument("json_path", type=str, help="Path to Json file containing annotations from CVAT")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    cvat_to_openpcdet(argdict["pcd_path"], argdict["json_path"])