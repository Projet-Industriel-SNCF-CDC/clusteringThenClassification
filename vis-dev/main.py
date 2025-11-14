from engine import Engine
from controls import UIPanel
from engine import pygame
from models import Obstacle
from obstacles import ObstacleProvider
from point_provider import PointCloudProvider
from points_processing import PointCloudProcessor

if __name__ == "__main__":
    pygame.init()

    ui_panel = UIPanel(20, 20, 500, 350)

    slider_point_size = ui_panel.add_slider("Trim Left", 0, 10.0, 2, (255, 100, 100))
    slider_trim_right = ui_panel.add_slider("Trim Right", 0, 10.0, 2, (100, 100, 255))
    slider_trim_bottom = ui_panel.add_slider("Trim Bottom", -4, -1, -4, (100, 255, 100))
    slider_adjust_bottom_trim_angle = ui_panel.add_slider(
        "Bottom Trim Angle", -3, 3, 0, (255, 255, 100)
    )

    point_provider = PointCloudProvider(data_path="./data/rails/")
    # x1, y1, x2, y2
    obstacle_provider = ObstacleProvider(data_path="./data/obstacles/", obstacle_box=(-1, -5, 1, 15))
    obstacle_provider.apply_tranform(Obstacle.transform, 0.3, [0, -1.1, -10])
    obstacle_provider.apply_tranform(Obstacle.rotate, 3.14 / 2, 0, 0)

    point_processor = PointCloudProcessor()
    engine = Engine(
        ui_panel=ui_panel,
        point_provider=point_provider,
        point_processor=point_processor,
        obstacle_provider=obstacle_provider,
    )
    engine.run()
