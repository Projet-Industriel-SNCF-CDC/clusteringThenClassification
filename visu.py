import pandas as pd
import numpy as np
import plotly.graph_objects as go

def visualize_pointcloud_csv1(csv_file_path, sample_size=None):
    """
    Read and visualize a CSV point cloud file in 3D.

    Args:
        csv_file_path (str): Path to the CSV file
        sample_size (int, optional): Number of points to sample. Use None to plot all points.
    """
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Make sure it has x, y, z columns
    if not all(col in df.columns for col in ['x', 'y', 'z']):
        raise ValueError("CSV file must contain columns: x, y, z")

    points = df[['x', 'y', 'z']].to_numpy()

    # Sample points if specified
    if sample_size and sample_size < len(points):
        indices = np.random.choice(len(points), sample_size, replace=False)
        points = points[indices]

    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 1],  # y-axis
        y=points[:, 0],  # x-axis
        z=points[:, 2],  # z-axis
        mode='markers',
        marker=dict(
            size=1,
            opacity=0.8,
            color=points[:, 2],  # color by z
            colorscale='jet',
            showscale=True
        )
    )])

    fig.update_layout(
        scene=dict(
            aspectmode='data'
        ),
        width=1000,
        height=800
    )

    fig.show()



# Run the code
#visualize_pointcloud_csv("rails/lidar_points_0.0.csv")
if __name__ == "__main__":
    #visualize_pointcloud_csv1("rails/lidar_points_with_labels.csv")
    visualize_pointcloud_csv1("export/cluster_0.csv")
    visualize_pointcloud_csv1("export/cluster_1.csv")
    visualize_pointcloud_csv1("export/cluster_2.csv")
    visualize_pointcloud_csv1("export/cluster_3.csv")
    visualize_pointcloud_csv1("export/cluster_4.csv")
    visualize_pointcloud_csv1("export/cluster_6.csv")
    visualize_pointcloud_csv1("export/cluster_7.csv")
