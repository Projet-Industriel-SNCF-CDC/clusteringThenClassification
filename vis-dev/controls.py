import pygame


class Slider:
    def __init__(
        self,
        x,
        y,
        width,
        height,
        min_val,
        max_val,
        initial_val,
        label,
        color=(100, 150, 255),
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.color = color
        self.dragging = False

        # Calculate handle position
        self.handle_radius = height // 2 + 2
        self.update_handle_pos()

    def update_handle_pos(self):
        """Update handle position based on current value"""
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        self.handle_x = self.x + int(ratio * self.width)

    def set_value(self, value):
        """Set slider value and clamp to range"""
        self.value = max(self.min_val, min(self.max_val, value))
        self.update_handle_pos()

    def set_value_from_pos(self, mouse_x):
        """Set value based on mouse position"""
        ratio = (mouse_x - self.x) / self.width
        ratio = max(0, min(1, ratio))
        self.value = self.min_val + ratio * (self.max_val - self.min_val)
        self.update_handle_pos()

    def is_handle_clicked(self, mouse_x, mouse_y):
        """Check if mouse is over the handle"""
        dx = mouse_x - self.handle_x
        dy = mouse_y - (self.y + self.height // 2)
        return dx * dx + dy * dy <= self.handle_radius * self.handle_radius

    def is_track_clicked(self, mouse_x, mouse_y):
        """Check if mouse is over the track"""
        return (
            self.x <= mouse_x <= self.x + self.width
            and self.y <= mouse_y <= self.y + self.height
        )

    def draw(self, screen, font):
        """Draw the slider on a pygame surface"""
        # Draw label
        label_surface = font.render(self.label, True, (255, 255, 255))
        screen.blit(label_surface, (self.x, self.y - 25))

        # Draw value
        value_text = f"{self.value:.2f}"
        value_surface = font.render(value_text, True, (200, 200, 200))
        screen.blit(value_surface, (self.x + self.width + 10, self.y - 3))

        # Draw track background
        track_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(
            screen, (60, 60, 60), track_rect, border_radius=self.height // 2
        )

        # Draw filled portion
        filled_width = int(
            (self.value - self.min_val) / (self.max_val - self.min_val) * self.width
        )
        if filled_width > 0:
            filled_rect = pygame.Rect(self.x, self.y, filled_width, self.height)
            pygame.draw.rect(
                screen, self.color, filled_rect, border_radius=self.height // 2
            )

        # Draw handle
        handle_color = (255, 255, 255) if self.dragging else (220, 220, 220)
        pygame.draw.circle(
            screen,
            handle_color,
            (self.handle_x, self.y + self.height // 2),
            self.handle_radius,
        )
        pygame.draw.circle(
            screen,
            self.color,
            (self.handle_x, self.y + self.height // 2),
            self.handle_radius - 2,
        )


class Button:
    def __init__(
        self,
        x,
        y,
        width,
        height,
        text,
        color=(100, 150, 255),
        text_color=(255, 255, 255),
        hover_color=(130, 180, 255),
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color
        self.text_color = text_color
        self.hover_color = hover_color
        self.pressed_color = (80, 130, 235)

        self.is_hovered = False
        self.is_pressed = False
        self.font = pygame.font.Font(None, 28)

    def is_mouse_over(self, mouse_pos):
        """Check if mouse is over the button"""
        return (
            self.x <= mouse_pos[0] <= self.x + self.width
            and self.y <= mouse_pos[1] <= self.y + self.height
        )

    def handle_event(self, event, mouse_pos):
        """
        Handle mouse events. Returns True if button was clicked.
        """
        self.is_hovered = self.is_mouse_over(mouse_pos)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_hovered:
                self.is_pressed = True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.is_pressed and self.is_hovered:
                self.is_pressed = False
                return True  # Button was clicked
            self.is_pressed = False

        return False

    def draw(self, surface):
        """Draw the button on a pygame surface"""
        # Determine color based on state
        if self.is_pressed:
            current_color = self.pressed_color
        elif self.is_hovered:
            current_color = self.hover_color
        else:
            current_color = self.color

        # Draw button rectangle
        button_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(surface, current_color, button_rect, border_radius=8)

        # Draw border
        border_color = (200, 200, 200) if self.is_hovered else (150, 150, 150)
        pygame.draw.rect(surface, border_color, button_rect, 2, border_radius=8)

        # Draw text centered
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(
            center=(self.x + self.width // 2, self.y + self.height // 2)
        )
        surface.blit(text_surface, text_rect)


class UIPanel:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.sliders = []
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 32)
        self.visible = True

    def add_slider(self, label, min_val, max_val, initial_val, color=(100, 150, 255)):
        """Add a slider to the panel"""
        slider_y = self.y + 60 + len(self.sliders) * 60
        slider = Slider(
            self.x + 20,
            slider_y,
            self.width - 120,
            10,
            min_val,
            max_val,
            initial_val,
            label,
            color,
        )
        self.sliders.append(slider)
        return slider

    def registered(self, mouse_pos):
        """Check if mouse clicked inside the panel"""
        return (
            self.x <= mouse_pos[0] <= self.x + self.width
            and self.y <= mouse_pos[1] <= self.y + self.height
        )

    def handle_mouse_down(self, mouse_pos):
        """Handle mouse button down event"""
        if not self.visible:
            return None

        for slider in self.sliders:
            if slider.is_handle_clicked(mouse_pos[0], mouse_pos[1]):
                slider.dragging = True
                return slider
            elif slider.is_track_clicked(mouse_pos[0], mouse_pos[1]):
                slider.set_value_from_pos(mouse_pos[0])
                slider.dragging = True
                return slider
        return None

    def handle_mouse_up(self):
        """Handle mouse button up event"""
        for slider in self.sliders:
            slider.dragging = False

    def handle_mouse_motion(self, mouse_pos):
        """Handle mouse motion event"""
        if not self.visible:
            return None

        for slider in self.sliders:
            if slider.dragging:
                old_value = slider.value
                slider.set_value_from_pos(mouse_pos[0])
                if (
                    abs(slider.value - old_value) > 0.001
                ):  # Only return if value changed
                    return slider
        return None

    def draw(self, screen):
        """Draw the panel and all sliders"""
        if not self.visible:
            return

        # Draw panel background
        panel_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        panel_surface.fill((30, 30, 30, 230))
        screen.blit(panel_surface, (self.x, self.y))

        # Draw border
        pygame.draw.rect(
            screen, (80, 80, 80), (self.x, self.y, self.width, self.height), 2
        )

        # Draw title
        title = self.title_font.render("Parameters", True, (255, 255, 255))
        screen.blit(title, (self.x + 20, self.y + 15))

        # Draw all sliders
        for slider in self.sliders:
            slider.draw(screen, self.font)

    def get_state(self):
        return [(slider.label, slider.value) for slider in self.sliders]
