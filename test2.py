import numpy as np
import hdbscan
import pandas as pd
from visu2 import visualize_pointcloud_csv  # Import the Plotly visualization function
from visu import visualize_pointcloud_csv1



def add_random_clusters(points, n_clusters=3, cluster_sizes=[50, 80, 100], cluster_spread=[0.1, 0.2, 0.2], cluster_centers=None):
    """
    Add random clusters to a point cloud.
    
    Args:
        points (np.ndarray): Original point cloud (N, 3)
        n_clusters (int): Number of clusters to add
        cluster_sizes (list): Number of points per cluster
        cluster_spread (list): Standard deviation for cluster points
        cluster_centers (list of lists, optional): Specific centers for clusters
    Returns:
        np.ndarray: Point cloud with added clusters
    """
    clusters = []
    for i in range(n_clusters):
        size = cluster_sizes[i % len(cluster_sizes)]
        spread = cluster_spread[i % len(cluster_spread)]
        if cluster_centers is not None and i < len(cluster_centers):
            center = cluster_centers[i]
        else:
            # Random center within the bounding box of original points
            min_vals = points.min(axis=0)
            max_vals = points.max(axis=0)
            center = np.random.uniform(min_vals, max_vals)
        cluster = np.random.randn(size, 3) * spread + center
        clusters.append(cluster)
    
    # Combine original points and new clusters
    all_points = np.vstack([points] + clusters)
    return all_points

def remove_lower_layer(points, n_lowest, y_range=(-1.5, 1.5)):
    """
    Remove the n lowest points in Z (lower layer) and keep only points within a Y range.
    
    Args:
        points (np.ndarray): Nx3 array of points
        n_lowest (int): Number of points to remove at the bottom (lowest Z)
        y_range (tuple): Keep points with y between y_range[0] and y_range[1]
        
    Returns:
        np.ndarray: Filtered points
    """
    # Remove the n lowest points in Z
    sorted_indices = np.argsort(points[:, 2])
    keep_indices_z = sorted_indices[n_lowest:]
    points_z_filtered = points[keep_indices_z]
    
    # Keep only points within the specified Y range
    y_min, y_max = y_range
    y_mask = (points_z_filtered[:, 0] >= y_min) & (points_z_filtered[:, 0] <= y_max)
    filtered_points = points_z_filtered[y_mask]
    
    return filtered_points

def filter_points(points, z_range=(-0.8,3), y_range=(-2, 2)):
    """
    Keep only points within specified Z and Y ranges.

    Args:
        points (np.ndarray): Nx3 array of points (x, y, z)
        z_range (tuple or None): (zmin, zmax) to keep points within this Z range
        y_range (tuple): (ymin, ymax) to keep points within this Y range

    Returns:
        np.ndarray: Filtered points
    """
    filtered_points = points

    # Filter by Z range if provided
    zmin, zmax = z_range
    z_mask = (filtered_points[:, 2] >= zmin) & (filtered_points[:, 2] <= zmax)
    filtered_points = filtered_points[z_mask]

    # Filter by Y range
    ymin, ymax = y_range
    y_mask = (filtered_points[:, 0] >= ymin) & (filtered_points[:, 0] <= ymax)
    filtered_points = filtered_points[y_mask]

    x_mask = (filtered_points[:, 1] <= 80)
    filtered_points = filtered_points[x_mask]

    return filtered_points


# 1. Load point cloud data from CSV
point = np.loadtxt('rails/lidar_points_0.0.csv', delimiter=',', skiprows=1)

# Example: Add 3 random clusters
point = add_random_clusters(point, n_clusters=3, cluster_centers=[[0.3, 19, 0], [-0.7, 40, 0], [-0.7, 10, 0]])


# 4. Save points + labels to CSV for visualization
df = pd.DataFrame(point, columns=['x', 'y', 'z'])
df.to_csv('rails/lidar_points_0.1.csv', index=False)



points = filter_points(point)




print(f"Nombre total de points : {len(points)}")
print(f"Dimensions du nuage : {points.shape}")
print(f"Étendue X : [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
print(f"Étendue Y : [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
print(f"Étendue Z : [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")






# 2. Apply HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,    # Minimum cluster size
    min_samples=30,           # Robustness to noise
    metric='euclidean',
    #cluster_selection_epsilon=0.01  # distance max pour fusionner les clusters
)
labels = clusterer.fit_predict(points)




# 3. Analyze clustering results
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"\n=== Résultats ===")
print(f"Nombre de clusters détectés : {n_clusters}")
print(f"Points de bruit : {n_noise}")
print(f"Labels uniques : {set(labels)}")


for cluster_id in set(labels):
    if cluster_id == -1:
        continue  # on ignore le bruit

    # Récupération des points du cluster
    mask = labels == cluster_id
    cluster_points = points[mask]
    n_points = cluster_points.shape[0]
    print(f"Cluster {cluster_id} : {n_points} points")

    # Export seulement si le cluster est suffisamment grand
    if n_points > 10 and n_points < 200 :
        # Convertir en DataFrame
        df_cluster = pd.DataFrame(cluster_points, columns=['x', 'y', 'z'])
        df_cluster['label'] = cluster_id

        # Sauvegarde dans un fichier CSV distinct
        filename = f"export/cluster_{cluster_id}.csv"
        df_cluster.to_csv(filename, index=False)
        print(f"Exported: {filename}")


# 4. Save points + labels to CSV for visualization
df = pd.DataFrame(points, columns=['x', 'y', 'z'])
df['label'] = labels
df.to_csv('rails/lidar_points_with_labels.csv', index=False)

# 5. Visualize point cloud using Plotly
# You can sample if the cloud is too large, e.g., sample_size=100000
visualize_pointcloud_csv('rails/lidar_points_with_labels.csv')
visualize_pointcloud_csv1('rails/lidar_points_0.1.csv')
