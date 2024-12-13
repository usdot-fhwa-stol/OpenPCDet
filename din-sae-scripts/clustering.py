import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.cluster import KMeans

def main(x_step, y_step, timesteps, save_progress=False):
    """
    Run clustering
    Inputs:
        - x_step (float): The spacing used to generate simulated LiDAR points along the x-axis
        - y_step (float): The spacing used to generate simulated LiDAR points along the y-axis
        - timesteps (int): The number of LiDAR frames
        - save_progress (bool): Flag to save PNG images for each iteration of the clustering process
    """
    # Create simulate point cloud in elevation vs. azimuth plane
    points = generate_simulated_points(x_step, y_step, timesteps, (0, 2))
    # Initialize centroids using the frame with the most points
    cluster_centroids = get_largest_frame(points)
    # Combine point clouds across all frames into a single 2D array
    flattened_points = np.vstack(points)
    # Run Kmeans clustering to tune the centroids
    kmeans = determine_clusters(cluster_centroids, flattened_points, len(cluster_centroids))
    count = 0  # Variable used for creating visualizatons
    while True:
        # Determine the largest cluster radius size possible given the current centroids
        radius = determine_cluster_radius(kmeans)
        # Find the point furthest from a centroid
        furthest_point = get_furthest_point_from_cluster(flattened_points, kmeans)
        restart = False
        for i in range(timesteps):
            # If there is a point that doesn't fit in a cluster, add a centroid where that point is and re-run Kmeans
            if not check_all_points_have_cluster(points[i], kmeans, radius):
                new_centroids = np.vstack((kmeans.cluster_centers_, furthest_point))
                kmeans = determine_clusters(new_centroids, flattened_points, len(new_centroids))
                print("Adding centroid at (%f, %f) and re-running K-means" % (furthest_point[0], furthest_point[1]))
                print("Current cluster radius is %f" % (radius))
                print("Now there are %d centroids" % (len(kmeans.cluster_centers_)))
                # Restart loop because there was a point that didn't yet have a cluster
                restart = True
                break
        if not restart:
            break
        # Optional progress visualization
        if save_progress:
            fig, ax = plt.subplots(1, timesteps)
            ax[0].plot(points[i][:,0], points[i][:,1], 'b.', label="LiDAR points")
            ax[0].scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='*', color='r', label="Cluster Centers")
            for j in range(len(kmeans.cluster_centers_)):
                circle = Circle(kmeans.cluster_centers_[j], radius=radius, fill=False)
                ax[0].add_patch(circle)
            for i in range(1, timesteps):
                ax[i].plot(points[i][:,0], points[i][:,1], 'b.')
                ax[i].scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='*', color='r')
                for j in range(len(kmeans.cluster_centers_)):
                    circle = Circle(kmeans.cluster_centers_[j], radius=radius, fill=False)
                    ax[i].add_patch(circle)
            ax[0].set_ylabel("Elevation (deg)")
            for i in range(len(ax)):
                a = ax[i]
                a.set_xlim([0, 2])
                a.set_ylim([0, 2])
                a.set_xlabel("Azimuth (deg)")
                a.set_title("Time: %d" % (i))
                a.set_aspect('equal')
            fig.set_size_inches(10, 8)  # Adjust width and height as needed
            fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.71))
            plt.savefig("%04d.png" % (count))
            plt.close()
            count += 1
    # Check that clusters are mutually exclusive and all points belong to a cluster
    for i in range(timesteps):
        assert(check_clusters_are_mutually_exclusive(kmeans, radius))
        assert(check_all_points_have_cluster(points[i], kmeans, radius))
    # Show results
    fig, ax = plt.subplots(1, timesteps)
    ax[0].plot(points[i][:,0], points[i][:,1], 'b.', label="LiDAR points")
    ax[0].scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='*', color='r', label="Cluster Centers")
    for j in range(len(kmeans.cluster_centers_)):
        circle = Circle(kmeans.cluster_centers_[j], radius=radius, fill=False)
        ax[0].add_patch(circle)
    for i in range(1, timesteps):
        ax[i].plot(points[i][:,0], points[i][:,1], 'b.')
        ax[i].scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='*', color='r')
        for j in range(len(kmeans.cluster_centers_)):
            circle = Circle(kmeans.cluster_centers_[j], radius=radius, fill=False)
            ax[i].add_patch(circle)
    ax[0].set_ylabel("Elevation (deg)")
    for i in range(len(ax)):
        a = ax[i]
        a.set_xlim([0, 2])
        a.set_ylim([0, 2])
        a.set_xlabel("Azimuth (deg)")
        a.set_title("Time: %d" % (i))
        a.set_aspect('equal')
    fig.set_size_inches(10, 8)  # Adjust width and height as needed
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.71))
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
    for i in range(len(kmeans.cluster_centers_)):
        for j in range(len(kmeans.cluster_centers_)):
            if np.linalg.norm(kmeans.cluster_centers_[i] - kmeans.cluster_centers_[j]) < diameter and i != j:
                diameter = np.linalg.norm(kmeans.cluster_centers_[i] - kmeans.cluster_centers_[j])
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
    for i in range(len(points)):
        num_clusters_belonging = 0
        for j in range(len(kmeans.cluster_centers_)):
            # If the point falls within the radius of a point that is not its assigned cluster, the clustering is not valid
            if np.linalg.norm(points[i] - kmeans.cluster_centers_[j]) <= radius:
                num_clusters_belonging += 1
        if num_clusters_belonging == 0:
            return False
    return True

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

if __name__ == "__main__":
    # Set random seed to keep results consistent
    np.random.seed(42)
    main(0.2, 0.2, 3, save_progress=True)