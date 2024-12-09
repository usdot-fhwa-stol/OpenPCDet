import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.cluster import KMeans

def main(num_points, timesteps):
    points = generate_simulated_points(num_points, timesteps, (0, 2))
    cluster_centroids = get_largest_frame(points)
    flattened_points = np.vstack(points)
    kmeans = determine_clusters(cluster_centroids, flattened_points, len(cluster_centroids))
    radius = determine_cluster_radius(flattened_points, kmeans, 0)
    clusters_passed = 0
    radius_shrink_iterations = 0
    while clusters_passed != timesteps:
        clusters_passed = 0
        for i in range(timesteps):
            if check_clusters_are_valid(points[i], kmeans, radius):
                clusters_passed += 1
            else:
                radius_shrink_iterations += 1
                radius = determine_cluster_radius(flattened_points, kmeans, radius_shrink_iterations)
    fig, ax = plt.subplots(1, timesteps)
    for i in range(timesteps):
        ax[i].plot(points[i][:,0], points[i][:,1], 'b.')
        ax[i].scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='*', color='r')
        for j in range(len(kmeans.cluster_centers_)):
            circle = Circle(kmeans.cluster_centers_[j], radius=radius, fill=False)
            ax[i].add_patch(circle)
    for a in ax:
        a.set_xlim([-0.1, 2.1])
        a.set_ylim([-0.1, 2.1])
        a.set_aspect('equal')
    plt.show()


"""
Generate a series of simulated lidar points on a plane within a specified range 
"""
def generate_simulated_points(num_points : int, 
                           timesteps: int,
                           square_range : tuple,
                           dropout_probability=0.05):
    points = []
    # Get initial points
    points.append(np.random.uniform(square_range[0], square_range[1], size=(num_points, 2)))
    # Add random noise for remaining frames
    for i in range(1, timesteps):
        points.append(points[0] + np.random.normal(loc=0, scale=0.05, size=(num_points, 2)))
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
    largest_frame = None
    largest_frame_size = 0
    for pc in points:
        if len(pc) > largest_frame_size:
            largest_frame_size = len(pc)
            largest_frame = pc
    return largest_frame


def determine_clusters(centroids, points, num_clusters):
    kmeans = KMeans(init=centroids, n_clusters=num_clusters, random_state=0, n_init='auto')
    kmeans.fit(points)
    return kmeans

def determine_cluster_radius(points, kmeans, iteration):
    distances = np.zeros(len(points))
    for i in range(len(points)):
        assigned_cluster = kmeans.labels_[i]
        distances[i] = np.linalg.norm(points[i] - kmeans.cluster_centers_[assigned_cluster])
    distances.sort()
    return distances[-1 - iteration]

"""
By definition of K-Means points will always be assigned to a cluster
Need to ensure that no points belong to two clusters
"""
def check_clusters_are_valid(points, kmeans, radius):
    for i in range(len(points)):
        num_clusters_belonging = 0
        for j in range(len(kmeans.cluster_centers_)):
            # If the point falls within the radius of a point that is not its assigned cluster, the clustering is not valid
            if np.linalg.norm(points[i] - kmeans.cluster_centers_[j]) <= radius:
                num_clusters_belonging += 1
        if num_clusters_belonging > 1:
            return False
    return True


if __name__ == "__main__":
    np.random.seed(42)
    main(50, 3)