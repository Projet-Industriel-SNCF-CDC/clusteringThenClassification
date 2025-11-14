from models import PointCloud
import numpy as np


class PointCloudExporter:
    @staticmethod
    def export(points: PointCloud, label_color: np.ndarray | None = None):
        flag = np.all(points.colors == 1.0, axis=1).astype(int)
        return np.column_stack((points.points, flag))
