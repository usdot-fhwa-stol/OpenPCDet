import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import os
from tqdm import tqdm
import argparse, argcomplete

def main(dataset_path, save_progress=False):
    """
    Run clustering
    Inputs:
        - dataset_path (str): The path to the LiDAR dataset (if none is provided, simulated points are generated)
        - save_progress (bool): Flag to save PNG images for each iteration of the clustering process
    """
    # Create simulate point cloud in elevation vs. azimuth plane
    if dataset_path:
        points = load_lidar_data(dataset_path)
    else:
        points = generate_simulated_points(0.1, 0.1, 3, (0, 2))
    timesteps = len(points)
    # Initialize centroids using the frame with the most points
    cluster_centroids = get_largest_frame(points)
    # Combine point clouds across all frames into a single 2D array
    flattened_points = np.vstack(points)
    # Get average spacing between points
    average_distance = get_average_distance_between_points(points[0])
    # Run Kmeans clustering to tune the centroids
    kmeans = determine_clusters(cluster_centroids, flattened_points, len(cluster_centroids))
    # new_centroids = merge_close_clusters(kmeans, average_distance * 0.8)
    # kmeans = determine_clusters(new_centroids, flattened_points, len(new_centroids))
    count = 0  # Variable used for creating visualizatons
    finished = False
    while not finished:
        # Determine the largest cluster radius size possible given the current centroids
        finished = True
        radius = determine_cluster_radius(kmeans)
        print("Cluster radius is %f" % (radius))
        # Find the point furthest from a centroid
        furthest_point = get_furthest_point_from_cluster(flattened_points, kmeans)
        # If there is a point that doesn't fit in a cluster, add a centroid where that point is and re-run Kmeans
        if not check_all_points_have_cluster(flattened_points, kmeans, radius):
            new_centroids = np.vstack((kmeans.cluster_centers_, furthest_point))
            kmeans = determine_clusters(new_centroids, flattened_points, len(new_centroids))
            print("Adding centroid at (%f, %f) and re-running K-means" % (furthest_point[0], furthest_point[1]))
            print("Now there are %d centroids" % (len(kmeans.cluster_centers_)))
            # Restart loop because there was a point that didn't yet have a cluster
            finished = False
        # Optional progress visualization
        print("Iteration:", count)
        if save_progress:
            fig, ax = plt.subplots()
            colors = plt.cm.viridis(np.linspace(0, 1, timesteps))
            ax.plot(points[0][:,0], points[0][:,1], '.', c=colors[0], label="LiDAR points")
            ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='*', color='r', label="Cluster Centers")
            for j in range(len(kmeans.cluster_centers_)):
                circle = Circle(kmeans.cluster_centers_[j], radius=radius, fill=False)
                ax.add_patch(circle)
            for i in range(1, timesteps):
                ax.plot(points[i][:,0], points[i][:,1], '.', c=colors[i])
            ax.set_ylabel("Elevation (deg)")
            ax.set_xlabel("Azimuth (deg)")
            ax.set_aspect('equal')
            ax.legend(loc="upper center", ncol=2)
            plt.savefig("%04d.png" % (count))
            plt.close()
        count += 1
    # Check that clusters are mutually exclusive and all points belong to a cluster
    print("Clusters are mutually exclusive:", check_clusters_are_mutually_exclusive(kmeans, radius))
    print("All points have a cluster:", check_all_points_have_cluster(flattened_points, kmeans, radius))
    # Show results
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, timesteps))
    ax.plot(points[0][:,0], points[0][:,1], '.', c=colors[0], label="LiDAR points")
    ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='*', color='r', label="Cluster Centers")
    for j in range(len(kmeans.cluster_centers_)):
        circle = Circle(kmeans.cluster_centers_[j], radius=radius, fill=False)
        ax.add_patch(circle)
    for i in range(1, timesteps):
        ax.plot(points[i][:,0], points[i][:,1], '.', c=colors[i])
    ax.set_ylabel("Elevation (deg)")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_aspect('equal')
    ax.legend(loc="upper center", ncol=2)
    plt.show()



def generate_simulated_points(x_step, y_step, timesteps, square_range, dropout_probability=0.05):
    """
    Generate a series of simulated lidar points on the elevation vs. azimuth plane within a specified range 
    Inputs:
        - x_step (float): The spacing used to generate simulated LiDAR points along the x-axis
        - y_step (float): The spacing used to generate simulated LiDAR points along the y-axis
        - timesteps (int): The number of LiDAR frames
        - square_range (tuple(int)): The range of azimuth and elevation points should be generated in
        - dropout_probability (float): The probability a LiDAR point is removed
    Output: list of np.ndarray, where each index in the list is a different LiDAR frame
    """
    points = []
    # Get initial points
    x_vals = np.arange(0, 2 + x_step, x_step) 
    y_vals = np.arange(0, 2 + y_step, y_step)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    points.append(np.vstack([x_grid.ravel(), y_grid.ravel()]).T)
    # Add random noise for remaining frames
    num_points = len(points[0])
    for i in range(1, timesteps):
        points.append(points[0] + np.random.normal(loc=0, scale=0.01, size=(num_points, 2)))
    # Add random noise to initial frame
    points[0] += np.random.normal(loc=0, scale=0.01, size=(num_points, 2))
    # Add random dropout for all frames
    for i in range(timesteps):
        indices_to_remove = []
        for j in range(len(points[i])):
            if points[i][j][0] < square_range[0] or points[i][j][1] < square_range[0] or points[i][j][0] > square_range[1] or points[i][j][1] > square_range[1]:
                indices_to_remove.append(j)
            elif np.random.rand() < dropout_probability:
                indices_to_remove.append(j)
        points[i] = np.delete(points[i], indices_to_remove, axis=0)
    return points

def get_largest_frame(points):
    """
    Get the LiDAR frame with the most points
    Inputs:
        - points (list(np.ndarray)): The simulated LiDAR points
    Output: The frame with the most points
    """
    largest_frame = None
    largest_frame_size = 0
    for pc in points:
        if len(pc) > largest_frame_size:
            largest_frame_size = len(pc)
            largest_frame = pc
    return largest_frame


def determine_clusters(centroids, points, num_clusters):
    """
    Run K-means clustering to refine cluster centroids
    Inputs:
        - centroids (np.ndarray): The initial centroids
        - points (np.ndarray): The LiDAR points
        - num_clusters (int): The number of cluster centroids
    Outputs: A SKLearn KMeans object with refined centroids

    Reference: https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.KMeans.html
    """
    kmeans = KMeans(init=centroids, n_clusters=num_clusters, random_state=0, n_init='auto')
    kmeans.fit(points)
    return kmeans

def get_furthest_point_from_cluster(points, kmeans):
    """
    Find the point that is furthest from a cluster centroid
    Inputs:
        - points (np.ndarray): The LiDAR points
        - kmeans (sklearn.cluster.KMeans): KMeans object with cluster centroids
    Output: The point furthest from a cluster centroid
    """
    max_distance = 0.0
    furthest_point = None
    for i in range(len(points)):
        assigned_cluster = kmeans.labels_[i]
        if np.linalg.norm(points[i] - kmeans.cluster_centers_[assigned_cluster]) > max_distance:
            max_distance = np.linalg.norm(points[i] - kmeans.cluster_centers_[assigned_cluster])
            furthest_point = points[i]
    return furthest_point

def determine_cluster_radius(kmeans):
    """
    Get the largest possible radius given the current cluster centroids
    This is equivalent to half the distance between the two closest cluster centroids
    to ensure clusters are mutually exclusive
    Inputs:
        - kmeans (sklearn.cluster.KMeans): KMeans object with cluster centroids
    Output: The cluster radius
    """
    diameter = np.inf
    kdtree = KDTree(kmeans.cluster_centers_)
    for i in range(len(kmeans.cluster_centers_)):
        distance, _ = kdtree.query(kmeans.cluster_centers_[i], k=2)
        if distance[1] < diameter:
            diameter = distance[1]
    return diameter / 2

def check_clusters_are_mutually_exclusive(kmeans, radius):
    """
    Check to ensure clusters are mutually exclusive
    Inputs:
        - kmeans (sklearn.cluster.KMeans): KMeans object with cluster radius
        - radius (float): Radius of each cluster
    Outputs: True if clusters are mutually exclusive, False if not
    """
    for i in range(len(kmeans.cluster_centers_)):
        for j in range(len(kmeans.cluster_centers_)):
            if np.linalg.norm(kmeans.cluster_centers_[i] - kmeans.cluster_centers_[j]) < 2*radius and i != j:
                return False
    return True

def check_all_points_have_cluster(points, kmeans, radius):
    """
    Check to ensure all LiDAR points belong to a cluster
    Inputs:
        - points (np.ndarray): The simulated LiDAR points
        - kmeans (sklearn.cluster.KMeans): KMeans object with cluster radius
        - radius (float): Radius of each cluster
    Outputs: True if clusters are mutually exclusive, False if not
    """
    points_without_cluster = 0
    for i in range(len(points)):
        assigned_cluster = kmeans.labels_[i]
        if np.linalg.norm(points[i] - kmeans.cluster_centers_[assigned_cluster]) > radius:
            points_without_cluster += 1
    print("There are %d points without a cluster" % (points_without_cluster))
    return points_without_cluster == 0

def get_average_number_of_detections(points, kmeans, radius, timesteps):
    """
    Get the average number of detections for each cluster
    Inputs:
        - points (np.ndarray): The simulated LiDAR points
        - kmeans (sklearn.cluster.KMeans): KMeans object with cluster radius
        - radius (float): Radius of each cluster
        - timesteps (int): The number of timesteps
    Output: The average number of detections for each cluster
    """
    detection_count = np.zeros(len(kmeans.cluster_centers_))
    for i in range(len(kmeans.cluster_centers_)):
        for point in points:
            if np.linalg.norm(kmeans.cluster_centers_[i] - point) < radius:
                detection_count[i] += 1
    return detection_count / timesteps

def merge_close_clusters(kmeans, distance_threshold):
    """
    Merge clusters if they are less than a distance threshold away
    Inputs:
        - kmeans (sklearn.cluster.KMeans): KMeans object
        - distance_threshold (float): Distance threshold to merge clusters
    Output: New centroids with merged clusters
    """
    merged = True
    new_centroids = kmeans.cluster_centers_
    while merged:
        n = len(new_centroids)
        merged = False
        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(new_centroids[i] - new_centroids[j])
                if distance < distance_threshold:
                    average_point = np.mean((new_centroids[i], new_centroids[j]), axis=0)
                    new_centroids = np.delete(new_centroids, [i, j], axis=0)
                    new_centroids = np.vstack([new_centroids, average_point])
                    print("Merging clusters to form (%f, %f)" % (average_point[0], average_point[1]))
                    merged = True
                    break
            if merged:
                break
    return new_centroids

def get_average_distance_between_points(points):
    """
    Get the average distance between LiDAR points
    Inputs:
        - points (np.ndarray): LiDAR points
    Output: The average euclidean distance between each point and its closest neighbor
    """
    tree = KDTree(points)
    distances, _ = tree.query(points, k=2)
    closest_distances = distances[:, 1]
    return np.mean(closest_distances)

def load_lidar_data(dataset_path, frame_limit=np.inf):
    """
    Load LiDAR data from a directory and convert it to azimuth/elevation
    Inputs:
        - dataset_path (str): A path to a directory containing KITTI-formatted data
        - frame_limit (int): The number of frames to load (if unspecified, all frames will be loaded)
    Output: The LiDAR data in azimuth vs. elevation format
    """
    lidar_files = [file for file in os.listdir(os.path.join(dataset_path, "velodyne"))]
    # Get the bounding box of the annotation target (Should be in the first annotation in the dataset)
    annotation_file = sorted(file for file in os.listdir(os.path.join(dataset_path, "label_2")))[0]
    with open(os.path.join(dataset_path, "label_2", annotation_file)) as f:
        annotation = f.readline().strip().split()
    length, width, height, x, y, z, yaw = np.asarray(annotation[8:], dtype=float)
    kitti_rotation = R.from_euler("zyx", (0.0, 90.0, -90.0), degrees=True)
    rotated_center = kitti_rotation.apply([x, y, z])
    # Rotation matrix for yaw angle of the bounding box
    yaw_rotation = R.from_euler('z', yaw)
    # Store points that fall within the first engineering target annotated
    point_clouds = []
    iterations = 0
    for lidar_file in tqdm(lidar_files, total=min(len(lidar_files), frame_limit)):
        if iterations == frame_limit:
            break
        lidar_file_path = os.path.join(dataset_path, "velodyne", lidar_file)
        # Load point cloud and remove intensity field
        point_cloud = np.delete(np.fromfile(lidar_file_path, dtype=np.float32).reshape(-1, 4), 3, axis=1)
        # Translate point cloud relative to the bounding box center
        translated_points = point_cloud - rotated_center
        # Rotate points using the yaw angle of the bounding box
        rotated_points = yaw_rotation.apply(translated_points)
         # Compute bounding box boundaries in local frame
        x_min, x_max = -width , width
        y_min, y_max = -length / 2 + 0.2, length / 2 - 0.2
        z_min, z_max = -height / 2 + 0.2, height / 2 - 0.2
        # Filter points within the boundaries
        mask = (
            (rotated_points[:, 0] >= x_min) & (rotated_points[:, 0] <= x_max) &
            (rotated_points[:, 1] >= y_min) & (rotated_points[:, 1] <= y_max) &
            (rotated_points[:, 2] >= z_min) & (rotated_points[:, 2] <= z_max)
        )
        point_clouds.append(cartesian_to_spherical(point_cloud[mask]))
        iterations += 1
    return point_clouds

def cartesian_to_spherical(points):
    """
    Convert points cartesian coordinates to spherical (azimuth + elevation)
    Inputs:
        - points (np.ndarray): LiDAR points in xyz format
    Output: LiDAR points in azimuth + elevation format
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Compute azimuth (lateral angle in XY-plane)
    azimuth = np.arctan2(y, x)

    # Compute elevation (vertical angle from the XY-plane)
    elevation = np.arctan2(z, x)

    # Combine results into a single array
    spherical_coords = np.column_stack((azimuth, elevation))
    return spherical_coords


if __name__ == "__main__":
    # Set random seed to keep results consistent
    np.random.seed(42)
    parser = argparse.ArgumentParser(description="Perform clustering according to the DIN SAE Spec on LiDAR data")
    parser.add_argument("--dataset_path", default="", help="Path to an annotated dataset")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    main(argdict["dataset_path"], save_progress=True)