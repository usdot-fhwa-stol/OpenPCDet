import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import os
from tqdm import tqdm
import argparse, argcomplete
from clustering import load_lidar_data

def main(dataset_path):
    """
    Visualize LiDAR scans over time
    Inputs:
        - dataset_path (str): The path to the LiDAR dataset (if none is provided, simulated points are generated)
    """
    points = load_lidar_data(dataset_path)
    timesteps = len(points)
    x_min, x_max, y_min, y_max = get_plot_dimensions(points)
    # Visualize point clouds
    for i in tqdm(range(timesteps)):
        fig, ax = plt.subplots()
        ax.plot(points[i][:,0], points[i][:,1], '.', label="LiDAR points")
        ax.set_ylabel("Elevation (deg)")
        ax.set_xlabel("Azimuth (deg)")
        ax.set_aspect('equal')
        ax.legend(loc="upper center", ncol=2)
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.savefig("%04d.png" % (i))
        plt.close()

def get_plot_dimensions(points):
    """
    Find the maximum and minimum azimuth and elevation in a series of LiDAR points
    Inputs:
        - points (np.ndarray): A time series of LiDAR points
    Output: min azimuth, max azimuth, min elevation, max elevation
    """
    flattened_points = np.vstack(points)
    minimums = np.min(flattened_points, axis=0)
    maximums = np.max(flattened_points, axis=0)
    return minimums[0], maximums[0], minimums[1], maximums[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LiDAR data on engineering targets over time")
    parser.add_argument("dataset_path", type=str, help="Path to an annotated dataset")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    main(argdict["dataset_path"])