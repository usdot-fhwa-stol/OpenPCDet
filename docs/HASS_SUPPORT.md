# HASS Support

## Installation

System Requirements:
1. Ubuntu 20.04
2. CUDA 11.8 Support

If not already installed, follow the instructions to install Miniconda [here](https://docs.anaconda.com/miniconda/miniconda-install/).
Once finished, follow these steps to install OpenPCDet.

1. Clone the repository with `git clone https://github.com/usdot-fhwa-stol/OpenPCDet.git`

2. Run `cd OpenPCDet/` followed by `conda env create --file=environment.yaml -n openpcdet` to create a conda virtual environment with all the necessary dependencies.

3. Run `conda activate openpcdet` to select your virtual environment and then `python3 setup.py develop` to install OpenPCDet.

## Demonstration

### Download Pretrained Models

Download pretrained models using the links provided on the main README, located [here](https://github.com/usdot-fhwa-stol/OpenPCDet?tab=readme-ov-file#kitti-3d-object-detection-baselines).

### Point Cloud Preparation

OpenPCDet requires point clouds be in either `.bin` or `.npy` format. To convert `.pcd` files to `.bin`, run

```
cd OpenPCDet/tools
python3 pcd2bin.py --pcd_path {Path to single PCD file or directory of PCD files} --bin_path {path to save BIN file or directory to save multiple BIN files}
```

Once you have your data as a bin file, you can run the OpenPCDet demo using:

```
cd OpenPCDet/tools
python3 demo.py --cfg_file cfgs/kitti_models/{model yaml file} --ckpt ../pretrained_models/{model weights} --data_path {path to LiDAR bin file}
```
