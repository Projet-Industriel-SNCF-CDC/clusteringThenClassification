import numpy as np
from models import Obstacle, PointCloud, PointOperationType

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class PointCloudProcessor:
    def __init__(self):
        self.trim_angle = 0.0
        self.ray_origin = (0.0, 0.0, 0.0)

    def apply_operations(
        self, point_cloud: PointCloud, operations: list[tuple[str, float]]
    ) -> PointCloud:
        points = point_cloud.points

        for key, value in operations:
            operation = key.lower()

            match operation:
                case PointOperationType.TRIM_LEFT:
                    points = points[points[:, 0] >= -value]
                case PointOperationType.TRIM_RIGHT:
                    points = points[points[:, 0] <= value]
                case PointOperationType.BOTTOM_TRIM_ANGLE:
                    self.trim_angle = value
                case PointOperationType.TRIM_BOTTOM:
                    angle_rad = np.deg2rad(self.trim_angle)
                    adjusted_z = points[:, 2] + np.sin(angle_rad) * points[:, 1]
                    points = points[adjusted_z >= value]

        return PointCloud(
            points=points,
            colors=(
                point_cloud.colors[: len(points)]
                if point_cloud.colors is not None
                else None
            ),
            metadata=point_cloud.metadata,
        )

    def apply_height_gradient(
        self, point_cloud: PointCloud, threshold: float = 1.0
    ) -> PointCloud:
        points = point_cloud.points

        z_min = np.min(points[:, 2])
        z_max = np.max(points[:, 2])
        z_norm = (points[:, 2] - z_min) / (z_max - z_min)

        mask = points[:, 2] < z_min + threshold
        z_norm[mask] = 1.0

        colors = np.zeros((len(points), 3), dtype=np.float32)
        colors[:, 0] = z_norm
        colors[:, 1] = 1.0 - np.abs(z_norm - 0.5) * 2
        colors[:, 2] = 1.0 - z_norm

        return PointCloud(points=points, colors=colors, metadata=point_cloud.metadata)

    def apply_uniform_color(
        self,
        point_cloud: PointCloud,
        color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> PointCloud:
        points = point_cloud.points
        colors = np.tile(color, (len(points), 1)).astype(np.float32)

        return PointCloud(points=points, colors=colors, metadata=point_cloud.metadata)

    def apply_shadow_vectorized(
        self, points: PointCloud, polygons: np.ndarray
    ) -> PointCloud:
        """
        Apply shadows from multiple polygons to a point cloud.

        Args:
            points: PointCloud object with points of shape (N, 3)
            polygons: Array of shape (M, 3, 3) where M is number of polygons,
                     each polygon has 3 vertices with 3 coordinates

        Returns:
            PointCloud with shadowed points colored black
        """
        if polygons.ndim == 2:
            polygons = polygons[np.newaxis, ...]

        num_polygons = polygons.shape[0]
        num_points = len(points.points)

        # Initialize colors if needed
        if points.colors is None:
            points.colors = np.zeros_like(points.points)

        # Compute normals for all polygons at once: (M, 3)
        A = polygons[:, 0, :]  # (M, 3)
        B = polygons[:, 1, :]  # (M, 3)
        C = polygons[:, 2, :]  # (M, 3)

        normals = np.cross(B - A, C - A)  # (M, 3)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)  # (M, 1)
        normals = normals / norms  # (M, 3)

        # Compute plane equations
        plane_d = np.einsum("ij,ij->i", normals, A)  # (M,)

        # Vectorized ray-plane intersection for all polygons and points
        # denom: (M, N) = normals (M, 3) @ points.T (3, N)
        denom = np.dot(normals, points.points.T)  # (M, N)

        valid = denom != 0  # (M, N)

        # Compute t values: (M, N)
        t = np.zeros((num_polygons, num_points))
        t[valid] = (
            plane_d[:, np.newaxis].repeat(num_points, axis=1)[valid] / denom[valid]
        )

        valid &= t > 0

        # Compute intersection points X for all valid intersections
        # X: (M, N, 3)
        X = points.points[np.newaxis, :, :] * t[:, :, np.newaxis]  # (M, N, 3)

        # Barycentric coordinate test for all polygons
        v0 = C[:, np.newaxis, :] - A[:, np.newaxis, :]  # (M, 1, 3)
        v1 = B[:, np.newaxis, :] - A[:, np.newaxis, :]  # (M, 1, 3)
        v2 = X - A[:, np.newaxis, :]  # (M, N, 3)

        # Precompute dot products
        dot00 = np.einsum("mij,mij->m", v0, v0)  # (M,)
        dot01 = np.einsum("mij,mij->m", v0, v1)  # (M,)
        dot11 = np.einsum("mij,mij->m", v1, v1)  # (M,)

        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)  # (M,)

        # Compute for all points
        dot02 = np.einsum("mni,mi->mn", v2, v0.squeeze(1))  # (M, N)
        dot12 = np.einsum("mni,mi->mn", v2, v1.squeeze(1))  # (M, N)

        u = (dot11[:, np.newaxis] * dot02 - dot01[:, np.newaxis] * dot12) * invDenom[
            :, np.newaxis
        ]
        v = (dot00[:, np.newaxis] * dot12 - dot01[:, np.newaxis] * dot02) * invDenom[
            :, np.newaxis
        ]

        # Check if inside triangle: (M, N)
        inside = (u >= 0) & (v >= 0) & (u + v <= 1) & valid

        # Any polygon casting shadow on a point makes it black
        shadowed_points = np.any(inside, axis=0)  # (N,)

        points.colors[shadowed_points] = np.array([0.0, 0.0, 0.0])

        return points

    def apply_shadow(self, points: PointCloud, polygon: np.ndarray) -> PointCloud:
        A = polygon[0]
        B = polygon[1]
        C = polygon[2]
        n = np.cross(B - A, C - A)
        n = n / np.linalg.norm(n)

        num = np.dot(n, A)
        denom = np.dot(points.points, n)

        valid = denom != 0
        t = np.zeros(len(points.points))
        t[valid] = num / denom[valid]

        valid &= t > 0
        X = points.points * t[:, None]

        v0 = C - A
        v1 = B - A
        v2 = X - A

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot11 = np.dot(v1, v1)
        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)

        dot02 = np.einsum("ij,j->i", v2, v0)
        dot12 = np.einsum("ij,j->i", v2, v1)

        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        inside = (u >= 0) & (v >= 0) & (u + v <= 1) & valid
        if points.colors is None:
            points.colors = np.zeros_like(points.points)

        points.colors[inside] = np.array([0.0, 0.0, 0.0])

        return points

    def project_on_obstacle(
        self, points: PointCloud, polygon: np.ndarray
    ) -> PointCloud:
        A = polygon[0]
        B = polygon[1]
        C = polygon[2]

        n = np.cross(B - A, C - A)
        n = n / np.linalg.norm(n)

        num = np.dot(n, A)
        denom = np.dot(points.points, n)

        valid = denom != 0
        t = np.zeros(len(points.points))
        t[valid] = num / denom[valid]

        valid &= t > 0
        X = points.points * t[:, None]

        v0 = C - A
        v1 = B - A
        v2 = X - A

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot11 = np.dot(v1, v1)
        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)

        dot02 = np.einsum("ij,j->i", v2, v0)
        dot12 = np.einsum("ij,j->i", v2, v1)

        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        inside = (u >= 0) & (v >= 0) & (u + v <= 1) & valid

        projected_points = X[inside]

        new_points = np.vstack([points.points, projected_points])

        if points.colors is not None:
            white_color = np.ones((projected_points.shape[0], points.colors.shape[1]))
            new_colors = np.vstack([points.colors, white_color])
        else:
            new_colors = np.ones((new_points.shape[0], 3))

        return PointCloud(
            points=new_points, colors=new_colors, metadata=points.metadata.copy()
        )

    def apply_shadow_gpu(self, points: PointCloud, polygons: np.ndarray, delete_shadow=False) -> PointCloud:
        """GPU-accelerated version using CuPy."""
        if not HAS_CUPY:
            return self.apply_shadow(points, polygons)

        if polygons.ndim == 2:
            polygons = polygons[np.newaxis, ...]

        if points.colors is None:
            points.colors = np.zeros_like(points.points)

        # Transfer to GPU
        pts_gpu = cp.asarray(points.points)
        polys_gpu = cp.asarray(polygons)

        A = polys_gpu[:, 0, :]
        B = polys_gpu[:, 1, :]
        C = polys_gpu[:, 2, :]

        normals = cp.cross(B - A, C - A)
        norms = cp.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms

        plane_d = cp.einsum("ij,ij->i", normals, A)
        denom = cp.dot(normals, pts_gpu.T)

        valid = denom != 0
        t = cp.zeros((polys_gpu.shape[0], pts_gpu.shape[0]))
        t[valid] = (
            plane_d[:, cp.newaxis].repeat(pts_gpu.shape[0], axis=1)[valid]
            / denom[valid]
        )

        valid &= t > 0
        X = pts_gpu[cp.newaxis, :, :] * t[:, :, cp.newaxis]

        v0 = C[:, cp.newaxis, :] - A[:, cp.newaxis, :]
        v1 = B[:, cp.newaxis, :] - A[:, cp.newaxis, :]
        v2 = X - A[:, cp.newaxis, :]

        dot00 = cp.einsum("mij,mij->m", v0, v0)
        dot01 = cp.einsum("mij,mij->m", v0, v1)
        dot11 = cp.einsum("mij,mij->m", v1, v1)

        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)

        dot02 = cp.einsum("mni,mi->mn", v2, v0.squeeze(1))
        dot12 = cp.einsum("mni,mi->mn", v2, v1.squeeze(1))

        u = (dot11[:, cp.newaxis] * dot02 - dot01[:, cp.newaxis] * dot12) * invDenom[
            :, cp.newaxis
        ]
        v = (dot00[:, cp.newaxis] * dot12 - dot01[:, cp.newaxis] * dot02) * invDenom[
            :, cp.newaxis
        ]

        inside = (u >= 0) & (v >= 0) & (u + v <= 1) & valid
        shadowed_points = cp.any(inside, axis=0)

        # Transfer back to CPU
        shadowed_cpu = cp.asnumpy(shadowed_points)
        # points.points = points.points[~shadowed_cpu]
        # points.colors = points.colors[~shadowed_cpu]
        if not delete_shadow:
            points.colors[shadowed_cpu] = np.array([0.0, 0.0, 0.0])

            return points
        else:
            points.points = points.points[~shadowed_cpu]
            points.colors = points.colors[~shadowed_cpu]
            return points


    def project_on_obstacle_gpu(
        self, points: PointCloud, polygons: np.ndarray, return_projection=False
    ) -> PointCloud:
        """GPU-accelerated version using CuPy for multiple polygons."""
        if not HAS_CUPY:
            return self.project_on_obstacle(points, polygons)

        # Ensure polygons is 3D array (num_polygons, 3, 3)
        if polygons.ndim == 2:
            polygons = polygons[np.newaxis, ...]

        # Transfer to GPU
        pts_gpu = cp.asarray(points.points)
        polys_gpu = cp.asarray(polygons)

        A = polys_gpu[:, 0, :]  # (num_polygons, 3)
        B = polys_gpu[:, 1, :]  # (num_polygons, 3)
        C = polys_gpu[:, 2, :]  # (num_polygons, 3)

        # Calculate normal vectors for all polygons
        normals = cp.cross(B - A, C - A)  # (num_polygons, 3)
        norms = cp.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms

        # Calculate projections for all polygons and points
        num = cp.einsum("ij,ij->i", normals, A)  # (num_polygons,)
        denom = cp.dot(normals, pts_gpu.T)  # (num_polygons, num_points)

        valid = denom != 0
        t = cp.zeros((polys_gpu.shape[0], pts_gpu.shape[0]))
        t[valid] = (
            num[:, cp.newaxis].repeat(pts_gpu.shape[0], axis=1)[valid] / denom[valid]
        )

        valid &= t > 0
        X = (
            pts_gpu[cp.newaxis, :, :] * t[:, :, cp.newaxis]
        )  # (num_polygons, num_points, 3)

        # Barycentric coordinate calculation for all polygons
        v0 = C[:, cp.newaxis, :] - A[:, cp.newaxis, :]  # (num_polygons, 1, 3)
        v1 = B[:, cp.newaxis, :] - A[:, cp.newaxis, :]  # (num_polygons, 1, 3)
        v2 = X - A[:, cp.newaxis, :]  # (num_polygons, num_points, 3)

        dot00 = cp.einsum("mij,mij->m", v0, v0)  # (num_polygons,)
        dot01 = cp.einsum("mij,mij->m", v0, v1)  # (num_polygons,)
        dot11 = cp.einsum("mij,mij->m", v1, v1)  # (num_polygons,)

        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)

        dot02 = cp.einsum("mni,mi->mn", v2, v0.squeeze(1))  # (num_polygons, num_points)
        dot12 = cp.einsum("mni,mi->mn", v2, v1.squeeze(1))  # (num_polygons, num_points)

        u = (dot11[:, cp.newaxis] * dot02 - dot01[:, cp.newaxis] * dot12) * invDenom[
            :, cp.newaxis
        ]
        v = (dot00[:, cp.newaxis] * dot12 - dot01[:, cp.newaxis] * dot02) * invDenom[
            :, cp.newaxis
        ]

        # Check if points are inside triangles
        inside = (
            (u >= 0) & (v >= 0) & (u + v <= 1) & valid
        )  # (num_polygons, num_points)

        # Collect all projected points that are inside any polygon
        # Flatten to get all valid projected points
        projected_mask = cp.any(
            inside, axis=0
        )  # (num_points,) - points that project into at least one polygon

        # Get the projected points for each polygon where inside is True
        all_projected_points = []
        for poly_idx in range(polys_gpu.shape[0]):
            poly_inside = inside[poly_idx]
            if cp.any(poly_inside):
                projected_pts = X[poly_idx][poly_inside]
                all_projected_points.append(projected_pts)

        # Transfer back to CPU and concatenate
        if all_projected_points:
            projected_points = cp.vstack(all_projected_points)
            projected_points = cp.asnumpy(projected_points)
        else:
            projected_points = np.empty((0, 3))

        # Combine original and projected points
        if not return_projection:
            new_points = np.vstack([points.points, projected_points])

            if points.colors is not None:
                white_color = np.ones(
                    (projected_points.shape[0], points.colors.shape[1])
                )
                new_colors = np.vstack([points.colors, white_color])
            else:
                new_colors = np.ones((new_points.shape[0], 3))

            return PointCloud(
                points=new_points, colors=new_colors, metadata=points.metadata.copy()
            )
        else:
            # new_points = np.vstack([points.points, projected_points])

            white_color = np.ones(
                (projected_points.shape[0], 3)
            )
            return PointCloud(
                points=projected_points,
                colors=white_color,
                metadata=points.metadata.copy(),
            )
