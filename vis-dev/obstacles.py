from typing import List
from models import Obstacle
import numpy as np
import os
import random
import copy
import math


def load_obj_as_obstacle(file_path: str) -> Obstacle:
    vertices = []
    faces = []

    with open(file_path, "r") as f:
        if not f.name.endswith(".obj"):
            raise Exception("Wrong file format, only .obj files accepted")
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()[1:]
                vertices.append([float(p) for p in parts])
            elif line.startswith("f "):
                # OBJ indices are 1-based, may contain slashes for textures/normals
                parts = line.strip().split()[1:]
                face = [int(p.split("/")[0]) - 1 for p in parts]  # convert to 0-based
                # Only keep triangles
                if len(face) == 3:
                    faces.append(face)
                else:
                    # Triangulate polygons with more than 3 vertices (simple fan)
                    for i in range(1, len(face) - 1):
                        faces.append([face[0], face[i], face[i + 1]])

    vertices = np.array(vertices, dtype=float)
    polygons = np.array([[vertices[i] for i in face] for face in faces], dtype=float)
    return Obstacle(polygons=polygons, coordinates=np.array([0, 0, 0], dtype=float))


class ObstacleProvider:
    def __init__(self, data_path: str | None = None, obstacle_box=None) -> None:
        self._data_path = data_path
        self.obstacles: List[Obstacle] = []
        self._files_loaded: set[str] = set()
        self._obstacle_box = obstacle_box

        if data_path is not None:
            self._files_available: list[str] = (
                os.listdir(data_path) if data_path else []
            )

        for f in self._files_available:
            self.obstacles.append(
                load_obj_as_obstacle(os.path.join(self._data_path, f))
            )

    def get_obstacle(self) -> List[Obstacle]:
        if self._obstacle_box:
            x_left = self._obstacle_box[0]
            x_right = self._obstacle_box[2]
            y_near = self._obstacle_box[1]
            y_far = self._obstacle_box[3]

            shuffled_obstacles = copy.deepcopy(self.obstacles)
            for o in shuffled_obstacles:
                x_offset = random.uniform(x_left, x_right)
                y_offset = random.uniform(y_near, y_far)

                rot_z = random.uniform(-math.pi, math.pi )

                o.rotate(0, 0, rot_z, [0, -o.coordinates[2], 0])
                o.transform(1, [x_offset, y_offset, 0])
            return shuffled_obstacles

        return self.obstacles

    def apply_tranform(self, function, *args):
        for o in self.obstacles:
            function(o, *args)
