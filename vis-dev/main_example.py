from engine_stripped import Engine
from controls import UIPanel
from engine import pygame
from point_provider import DumbProvider
from points_processing import PointCloudProcessor
import numpy as np

if __name__ == "__main__":
    pygame.init()

    ui_panel = UIPanel(0, 0, 1, 1)
    #
    points = np.loadtxt("./data/rails/lidar_points_0.0.csv", delimiter=",", skiprows=1)
    point_provider = DumbProvider(points)

    point_processor = PointCloudProcessor()
    engine = Engine(
        ui_panel=ui_panel,
        point_provider=point_provider,
        point_processor=point_processor,
    )
    engine.run()
