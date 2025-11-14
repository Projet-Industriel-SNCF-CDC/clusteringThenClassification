import pandas as pd
import numpy as np
import plotly.graph_objects as go

def visualize_pointcloud_csv(csv_file_path, sample_size=None, label_column='label'):
    """
    Read and visualize a CSV point cloud file in 3D with optional cluster coloring.

    Args:
        csv_file_path (str): Path to the CSV file
        sample_size (int, optional): Number of points to sample. Use None to plot all points.
        label_column (str, optional): Column name for point labels/clusters. If not present, colors by Z.
    """
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Make sure it has x, y, z columns
    if not all(col in df.columns for col in ['x', 'y', 'z']):
        raise ValueError("CSV file must contain columns: x, y, z")

    # Sample points if specified
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    # Extract points
    points = df[['x', 'y', 'z']].to_numpy()

    # Determine coloring
    if label_column in df.columns:
        labels = df[label_column].to_numpy()
        unique_labels = np.unique(labels)
        # Map labels to colors
        colors = np.zeros_like(labels, dtype=int)
        for i, lbl in enumerate(unique_labels):
            colors[labels == lbl] = i
        color_scale = 'rainbow'  # Utiliser une colorscale valide pour les clusters
        showscale = False
    else:
        colors = points[:, 2]  # color by Z if no label
        color_scale = 'jet'
        showscale = True

    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 1],  # y-axis
        y=points[:, 0],  # x-axis
        z=points[:, 2],  # z-axis
        mode='markers',
        marker=dict(
            size=2,
            opacity=0.8,
            color=colors,
            colorscale=color_scale,
            showscale=showscale
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

if __name__ == "__main__":
    visualize_pointcloud_csv("rails/lidar_points_with_labels.csv")
