from copy import copy, deepcopy
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LESS,
    GL_MODELVIEW,
    GL_NICEST,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINT_SMOOTH,
    GL_POINT_SMOOTH_HINT,
    GL_PROJECTION,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_UNSIGNED_BYTE,
    glBlendFunc,
    glClear,
    glClearColor,
    glDepthFunc,
    glDisable,
    glEnable,
    glHint,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glTranslatef,
    glViewport,
)
from OpenGL.GLU import gluPerspective
from OpenGL.raw.GL.VERSION.GL_1_0 import glDrawPixels
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL

from controls import UIPanel
from export import PointCloudExporter
from models import Obstacle, PointCloud
from obstacles import ObstacleProvider, load_obj_as_obstacle
from point_provider import PointCloudProvider
from points_processing import PointCloudProcessor
from rendering import ObstacleRenderer, PointCloudRenderer, RayRenderer


class Camera:
    def __init__(self):
        self.distance = 5.0
        self.rotation_x = -90.0
        self.rotation_y = 0.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.fov = 45.0

    def apply_transformations(self):
        """Apply camera transformations"""
        glTranslatef(self.pan_x, self.pan_y, -self.distance)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 0, 1)

    def reset(self):
        """Reset camera to default position"""
        self.distance = 5.0
        self.rotation_x = -90.0
        self.rotation_y = 0.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.fov = 45.0


class Engine:
    def __init__(
        self,
        ui_panel: UIPanel | None,
        point_provider: PointCloudProvider | None,
        point_processor: PointCloudProcessor | None,
        obstacle_provider: ObstacleProvider | None = None,
        height: int = 800,
        width: int = 1280,
    ) -> None:
        # responsible for camera, managing the pipeline, calling point and object rendering
        self.running = True
        self.ui_panel = ui_panel
        self.ui_surface = (
            pygame.Surface((width, height), pygame.SRCALPHA) if ui_panel else None
        )

        self.point_provider = point_provider
        self.point_processor = point_processor
        self.direction = "forward"  # or "backward"
        self.camera = Camera()

        self.height = height
        self.width = width

        self.display = pygame.display.set_mode(
            (self.width, self.height), DOUBLEBUF | OPENGL | pygame.RESIZABLE
        )

        self.mouse_down = False
        self.mouse_button = 0
        self.last_mouse_pos = None

        self.point_renderer = PointCloudRenderer()
        self.ray_renderer = RayRenderer()

        self.obstacle_provider = obstacle_provider

        self.keys_pressed = {}

        pygame.display.set_caption("fu")

        self.init_opengl()

    def init_opengl(self):
        """Initialize OpenGL settings"""
        # Set up perspective
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.camera.fov, self.width / self.height, 0.1, 100.0)

        # Set up modelview
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        # Enable point smoothing for better looking points
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        # Enable blending for smooth points
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Set background color (dark gray)
        glClearColor(0.1, 0.1, 0.1, 1.0)

    def handle_resize(self, width: int, height: int):
        self.width = width
        self.height = height
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.camera.fov, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        self.ui_surface = (
            pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            if self.ui_panel
            else None
        )

    def handle_events(self):
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    self.running = False
                case pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if self.ui_panel and self.ui_panel.registered(mouse_pos):
                        self.ui_panel.handle_mouse_down(mouse_pos)
                    else:
                        self.mouse_down = True
                        self.mouse_button = event.button
                        self.last_mouse_pos = mouse_pos

                case pygame.MOUSEBUTTONUP:
                    self.mouse_down = False
                    self.mouse_button = None
                    if self.ui_panel:
                        self.ui_panel.handle_mouse_up()

                case pygame.MOUSEMOTION:
                    mouse_pos = pygame.mouse.get_pos()
                    if self.ui_panel and self.ui_panel.registered(mouse_pos):
                        self.ui_panel.handle_mouse_motion(mouse_pos)
                    else:
                        self.handle_mouse_motion(
                            *mouse_pos, mouse_button=self.mouse_button
                        )
                    self.last_mouse_pos = mouse_pos

                case pygame.MOUSEWHEEL:
                    if self.ui_panel and not self.ui_panel.registered(
                        pygame.mouse.get_pos()
                    ):
                        self.handle_mouse_wheel(event.y)

                case pygame.KEYDOWN:
                    if event.key not in self.keys_pressed:
                        self.keys_pressed[event.key] = 1

                    match event.key:
                        case pygame.K_ESCAPE:
                            self.running = False

                        case pygame.K_SPACE:
                            self.camera.reset()
                            print("[CAMERA] Reset camera")

                        case pygame.K_n:
                            self.direction = "forward"

                        case pygame.K_p:
                            self.direction = "backward"

                case pygame.KEYUP:
                    if event.key in self.keys_pressed:
                        del self.keys_pressed[event.key]

                case pygame.WINDOWRESIZED:
                    width = event.dict.get("x", self.width)
                    height = event.dict.get("y", self.width)
                    self.handle_resize(width, height)

                case pygame.VIDEORESIZE:
                    width, height = event.dict.get("size", (self.width, self.height))
                    self.handle_resize(width, height)

                case _:
                    pass

        if pygame.K_n in self.keys_pressed:
            self.keys_pressed[pygame.K_n] += 1
        if pygame.K_p in self.keys_pressed:
            self.keys_pressed[pygame.K_p] += 1
        if pygame.K_n in self.keys_pressed and self.keys_pressed[pygame.K_n] > 20:
            self.direction = "forward"
        if pygame.K_p in self.keys_pressed and self.keys_pressed[pygame.K_p] > 20:
            self.direction = "backward"

    def handle_mouse_motion(self, x: int, y: int, mouse_button: int):
        """Handle mouse movement"""
        dx = x - self.last_mouse_pos[0] if self.last_mouse_pos else 0
        dy = y - self.last_mouse_pos[1] if self.last_mouse_pos else 0

        if mouse_button == 1:  # Left button - rotate
            self.camera.rotation_y += dx * 0.5
            self.camera.rotation_x += dy * 0.5

        elif mouse_button == 3:  # Right button - pan
            sensitivity = 0.01 * self.camera.distance
            self.camera.pan_x += dx * sensitivity
            self.camera.pan_y -= dy * sensitivity

    def handle_mouse_wheel(self, y):
        """Handle mouse wheel for zoom"""
        zoom_speed = 1
        self.camera.distance -= y * zoom_speed

    def process_points(
        self, points: PointCloud | None, obstacles: list[Obstacle] | None = None
    ):
        p = PointCloudProcessor()
        points = p.apply_operations(points, self.control_state)
        points = p.apply_height_gradient(points)
        # obstacle_clouds = PointCloud(np.empty((0, 3)))
        # # we do 2 passes, because first we need to project points on the obstacle, and then delete the points in the shadow
        # if obstacles is not None:
        #     for obstacle in obstacles:
        #         obstacle.cull_invisible_polygons([0.0, 0.0, 0.0])
        #         # points = p.apply_shadow_gpu(points, obstacle.polygons, delete_shadow=True)
        #         obstacle = p.project_on_obstacle_gpu(
        #             points, obstacle.polygons, return_projection=True
        #         )
        #         obstacle_clouds.concat(obstacle)
        #         # points = p.project_on_obstacle_gpu(
        #         #     points, obstacle.polygons, return_projection=True
        #         # )
        #         pass
        #
        # if obstacles is not None:
        #     for obstacle in obstacles:
        #         obstacle.cull_invisible_polygons([0.0, 0.0, 0.0])
        #         points = p.apply_shadow_gpu(
        #             points, obstacle.polygons, delete_shadow=True
        #         )

        self.points = [points]

    def process_obstacles(self, obstacles: list[Obstacle] | None = None):
        self.obstacles = obstacles if obstacles is not None else []

    def render_2d(self):
        if not self.ui_surface:
            return
        self.ui_surface.fill((0, 0, 0, 0))

        if self.ui_panel:
            self.ui_panel.draw(self.ui_surface)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)

        texture_data = pygame.image.tostring(self.ui_surface, "RGBA", True)
        glDrawPixels(self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

        glEnable(GL_DEPTH_TEST)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # type :ignore
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.camera.apply_transformations()

        for point_set in self.points:
            renderer = PointCloudRenderer(point_set)
            renderer.render()
        # for point_set in self.points:
        #     renderer = RayRenderer(point_set, drop_percentage=0.995)
        #     renderer.render()

        # for obstacle in self.obstacles:
        #     obstacle_renderer = ObstacleRenderer(obstacle)
        #     obstacle_renderer.render()

        self.render_2d()
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()

        if self.obstacle_provider:
            obstacles = self.obstacle_provider.get_obstacle()
        else:
            obstacles = []
        ctr = 0
        # TODO: remove
        while self.running:
            self.handle_events()
            points = self.point_provider.get_point_cloud(self.direction)
            # points_for_saving = deepcopy(points)
            dir_backup = self.direction
            if self.direction != "current":
                if self.obstacle_provider:
                    obstacles = self.obstacle_provider.get_obstacle()
                else:
                    obstacles = []

            self.direction = "current"
            self.control_state = self.ui_panel.get_state() if self.ui_panel else {}
            self.process_points(points, obstacles)
            self.render()
            fps = clock.get_fps()
            print(f"FPS: {fps:.2f}", end="\r")
            clock.tick(120)
