import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.cluster import KMeans

def main(num_points, timesteps):
    print("It's clustering time")
    points = generate_simulated_points(num_points, timesteps, (0, 2))
    flattened_points = points.reshape(-1, 2)
    kmeans = determine_clusters(points[0], flattened_points, num_points)
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
        print(clusters_passed)
    fig, ax = plt.subplots(1, timesteps)
    for i in range(timesteps):
        ax[i].plot(points[i,:,0], points[i,:,1], 'b.')
        ax[i].scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='*', color='r')
        for j in range(len(kmeans.cluster_centers_)):
            circle = Circle(kmeans.cluster_centers_[j], radius=radius, fill=False)
            ax[i].add_patch(circle)
    for a in ax:
        a.set_xlim([0, 2])
        a.set_ylim([0, 2])
        a.set_aspect('equal')
    plt.show()


"""
Generate a series of simulated lidar points on a plane within a specified range 
"""
def generate_simulated_points(num_points : int, 
                           timesteps: int,
                           square_range : tuple):
    points = np.zeros((timesteps, num_points, 2))
    # Get initial points
    points[0] = np.random.uniform(square_range[0], square_range[1], size=(num_points, 2))
    # Add random noise and random dropout for remaining frames
    # TODO: Simulate dropout and replaces points outside of range with infs
    for i in range(1, timesteps):
        points[i] = points[0] + np.random.normal(loc=0, scale=0.10, size=(num_points, 2))
    return points

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
    np.random.seed(0)
    main(50, 3)