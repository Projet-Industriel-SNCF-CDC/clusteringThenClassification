import os
from typing import Literal
from natsort import natsorted
import numpy as np

from models import PointCloud


class PointCloudProvider:
    """Manages a collection of PointCloud objects with navigation capabilities."""

    def __init__(
        self, point_cloud: PointCloud | None = None, data_path: str | None = None
    ) -> None:
        """
        Initialize the provider with an optional initial PointCloud.

        Args:
            point_cloud: Initial PointCloud to add to the collection
        """

        self._point_clouds: list[PointCloud] = []
        self._current_index: int = 0
        self._files_loaded = set()

        if data_path is not None:
            self._data_path = data_path

            self._files_available: list[str] = os.listdir(data_path) if data_path else []
            self._files_available = natsorted(self._files_available, reverse=True)


        if point_cloud is not None:
            self._point_clouds.append(point_cloud)

    def add_point_cloud(self, point_cloud: PointCloud) -> None:
        self._point_clouds.append(point_cloud)

    def get_point_cloud(
        self, direction: Literal["forward", "backward", "current"]
    ) -> PointCloud | None:
        if direction == "current":
            if 0 <= self._current_index < len(self._point_clouds):
                return self._point_clouds[self._current_index]
            return None
        if direction == "forward":
            return self._get_next_point_cloud()
        elif direction == "backward":
            return self._get_previous_point_cloud()
        else:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'forward' or 'backward'."
            )

    def _get_next_point_cloud(self) -> PointCloud | None:
        """Get the next PointCloud and advance the index."""
        if self._current_index >= len(self._point_clouds) - 2:
            if len(self._files_available) > 0:
                next_file = self._files_available.pop()
                self.load_points(os.path.join(self._data_path, next_file))

        point_cloud = self._point_clouds[self._current_index]
        if self._current_index < len(self._point_clouds) - 1:
            self._current_index += 1
        return point_cloud

    def _get_previous_point_cloud(self) -> PointCloud | None:
        """Get the previous PointCloud and move the index back."""

        if self._current_index > 0:
            self._current_index -= 1
        return self._point_clouds[self._current_index]

    def reset(self) -> None:
        """Reset the navigation index to the beginning."""
        self._current_index = 0

    def has_next(self) -> bool:
        """Check if there's a next PointCloud available."""
        return self._current_index < len(self._point_clouds)

    def has_previous(self) -> bool:
        """Check if there's a previous PointCloud available."""
        return self._current_index > 0

    @property
    def current_index(self) -> int:
        """Get the current navigation index."""
        return self._current_index

    @property
    def total_count(self) -> int:
        """Get the total number of PointClouds in the collection."""
        return len(self._point_clouds)

    def __len__(self) -> int:
        """Return the number of PointClouds in the collection."""
        return len(self._point_clouds)

    def __repr__(self) -> str:
        return f"PointCloudProvider(count={len(self._point_clouds)}, current_index={self._current_index})"

    
    def load_points(self, file_path: str):
        if file_path in self._files_loaded:
            return
    
        if file_path.endswith(".xyz"):
            points = np.loadtxt(file_path)
    
        elif file_path.endswith(".csv"):
            points = np.loadtxt(file_path, delimiter=",", skiprows=1)
    
        else:
            return
    
        self._files_loaded.add(file_path)
        self.add_point_cloud(PointCloud(points))

class DumbProvider:
    def __init__(self, points: np.ndarray) -> None:
        self.points = PointCloud(points)
    def get_point_cloud(self, *args, **kwargs):
        return self.points

