from OpenGL.GL import (
    GL_COLOR_ARRAY,
    GL_FLOAT,
    GL_LINE_LOOP,
    GL_LINES,
    GL_POINTS,
    GL_VERTEX_ARRAY,
    glColorPointer,
    glDisableClientState,
    glDrawArrays,
    glEnableClientState,
    glPointSize,
    glVertexPointer,
)
import numpy as np

from models import Obstacle, PointCloud


class PointCloudRenderer:
    def __init__(
        self, points: PointCloud | None = None, point_size: float = 5.0
    ) -> None:
        if points is None:
            return
        self.points = points.points
        self.colors = points.colors
        self.point_size = 5.0  # Default point size

    def set_points(self, points: PointCloud):
        self.points = points.points
        self.colors = points.colors

    def render(self):
        """Render the point cloud using vertex arrays (fast path)"""
        glPointSize(self.point_size)

        # Enable client states
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        # Provide vertex and color pointers
        glVertexPointer(3, GL_FLOAT, 0, self.points)
        glColorPointer(3, GL_FLOAT, 0, self.colors)

        # Draw all points in one call
        glDrawArrays(GL_POINTS, 0, len(self.points))

        # Disable client states
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)


class RayRenderer:
    def __init__(
        self, points: PointCloud | None = None, ray_origin=None, drop_percentage=0.9
    ) -> None:
        self.ray_origin = ray_origin if ray_origin is not None else (0.0, 0.0, 0.0)
        if points is None:
            return

        np.random.seed(42)
        n_keep = int(len(points.points) * (1 - drop_percentage))
        indices = np.random.choice(len(points.points), n_keep, replace=False)
        self.points = points.points[indices]
        self._prepare_ray_data()

    def set_points(self, points: PointCloud):
        self.points = points.points
        self._prepare_ray_data()

    def _prepare_ray_data(self):
        """Prepare vertex data for rays (origin to each point)"""
        num_points = len(self.points)

        # Create array for ray vertices: each ray has 2 vertices (origin and end point)
        self.ray_vertices = np.zeros((num_points * 2, 3), dtype=np.float32)

        # Fill in vertices: alternating origin and point
        for i in range(num_points):
            self.ray_vertices[i * 2] = self.ray_origin  # Ray start
            self.ray_vertices[i * 2 + 1] = self.points[i]  # Ray end

        # Create colors for rays (optional: fade from one color at origin to another at end)
        self.ray_colors = np.zeros((num_points * 2, 3), dtype=np.float32)
        for i in range(num_points):
            self.ray_colors[i * 2] = [1.0, 1.0, 1.0]  # White at origin
            self.ray_colors[i * 2 + 1] = [0.5, 0.5, 0.5]  # Gray at end point

    def render(self):
        # draw rays from origin to each point
        """Draw rays from origin to each point"""
        # Enable client states
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        # Provide vertex and color pointers
        glVertexPointer(3, GL_FLOAT, 0, self.ray_vertices)
        glColorPointer(3, GL_FLOAT, 0, self.ray_colors)

        # Draw all rays in one call
        glDrawArrays(GL_LINES, 0, len(self.ray_vertices))

        # Disable client states
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

class ObstacleRenderer:
    def __init__(self, obstacle: Obstacle) -> None:
        self.polygons = obstacle.polygons

    def render(self):
        for polygon in self.polygons:
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, polygon)
            glDrawArrays(GL_LINE_LOOP, 0, len(polygon))  # or GL_TRIANGLES for filled
            glDisableClientState(GL_VERTEX_ARRAY)
