from dataclasses import dataclass, field
from numpy import dtype, ndarray
from enum import Enum
import numpy as np


@dataclass
class PointCloud:
    points: ndarray
    colors: ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def concat(self, points):
        self.points = np.vstack([self.points, points.points])
        if self.colors is not None and points.colors is not None:
            self.colors = np.vstack([self.colors, points.colors])
        else:
            self.colors = np.ones((self.points.shape[0], 3))
        return self


class PointOperationType(str, Enum):
    TRIM_LEFT = "trim left"
    TRIM_RIGHT = "trim right"
    TRIM_BOTTOM = "trim bottom"
    BOTTOM_TRIM_ANGLE = "bottom trim angle"


@dataclass
class Obstacle:
    polygons: ndarray  # Array of shape (N, M, 3) representing N polygons with 3 vertices in 3D space
    coordinates: ndarray

    def transform(self, scale: float = 1.0, translate: np.ndarray = None):
        """
        Scale and translate the obstacle in place.
        :param scale: Uniform scale factor.
        :param translate: Optional 3D translation vector.
        """
        self.polygons *= scale
        if translate is not None:
            self.coordinates += translate
            translate = np.array(translate, dtype=float)
            self.polygons += translate

    def rotate(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, pivot=None):
        """
        Rotate the obstacle around X, Y, Z axes, with an optional pivot point.
        :param x: Rotation around X axis in radians.
        :param y: Rotation around Y axis in radians.
        :param z: Rotation around Z axis in radians.
        :param pivot: Optional pivot point (3D vector). If None, rotate around origin.
        """
        # Convert pivot to ndarray
        if pivot is not None:
            pivot = np.array(pivot, dtype=float)
    
        # Rotation matrices
        Rx = np.array([
            [1, 0,          0         ],
            [0, np.cos(x), -np.sin(x)],
            [0, np.sin(x),  np.cos(x)]
        ])
    
        Ry = np.array([
            [ np.cos(y), 0, np.sin(y)],
            [ 0,         1, 0        ],
            [-np.sin(y), 0, np.cos(y)]
        ])
    
        Rz = np.array([
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z),  np.cos(z), 0],
            [0,          0,         1]
        ])
    
        # Combined matrix (Z * Y * X)
        R = Rz @ Ry @ Rx
    
        # Translate to pivot → rotate → translate back
        if pivot is not None:
            self.polygons -= pivot
    
        self.polygons = self.polygons @ R.T
    
        if pivot is not None:
            self.polygons += pivot

    def cull_invisible_polygons(self, camera_position: np.ndarray, 
                              epsilon: float = 1e-6,
                              use_all_vertices: bool = True):
        """
        Remove polygons that are not visible from the given camera position.
        
        :param camera_position: 3D position of the camera [x, y, z]
        :param epsilon: Small value for numerical stability in comparisons
        :param use_all_vertices: If True, uses all vertices for center calculation, 
                               otherwise uses only first 3
        """
        camera_position = np.array(camera_position, dtype=float)
        
        # Vectorized version for better performance
        if len(self.polygons) == 0:
            return
        
        # Get vertices (assuming at least 3 vertices per polygon)
        v0 = self.polygons[:, 0, :]
        v1 = self.polygons[:, 1, :]
        v2 = self.polygons[:, 2, :]
        
        # Calculate normals for all polygons at once
        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = np.cross(edge1, edge2)
        
        # Calculate centers of polygons
        if use_all_vertices:
            centers = np.mean(self.polygons, axis=1)
        else:
            centers = (v0 + v1 + v2) / 3.0
        
        # Calculate view vectors from polygon centers to camera
        view_vectors = camera_position - centers
        
        # Check visibility using dot product
        # Positive dot product means polygon faces the camera
        dot_products = np.sum(normals * view_vectors, axis=1)
        visible_mask = dot_products > epsilon
        
        # Update polygons
        self.polygons = self.polygons[visible_mask]
    def reduce_polygons(self):
        pass
