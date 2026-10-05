"""Futuristic Laboratory Theme and Scientific UI Styling (§41)."""
from __future__ import annotations

from typing import Optional, Tuple
import pygame

# --- Dark Scientific Laboratory Palette ---
BG_DARK = (10, 14, 22)             # Void backdrop
BG_PANEL = (15, 22, 34, 225)       # Glass panel base
BG_PANEL_HEADER = (22, 32, 48, 240)
BORDER_DEFAULT = (40, 58, 82)      # Subtle panel border
BORDER_GLOW = (0, 220, 255)        # Electric cyan accent
BORDER_ACCENT = (14, 165, 233)

TEXT_PRIMARY = (240, 246, 252)     # Crisp white
TEXT_SECONDARY = (140, 165, 195)   # Muted scientific blue-gray
TEXT_MUTED = (85, 105, 130)        # Dim labels
TEXT_ACCENT = (56, 189, 248)       # Neon cyan
TEXT_SUCCESS = (74, 222, 128)      # Emerald
TEXT_WARNING = (251, 146, 60)      # Amber
TEXT_DANGER = (248, 113, 113)      # Crimson
TEXT_PURPLE = (192, 132, 252)      # Violet

BTN_NORMAL = (24, 36, 54)
BTN_HOVER = (36, 52, 78)
BTN_ACTIVE = (14, 116, 144)
BTN_BORDER = (45, 68, 98)


class Theme:
    def __init__(self):
        pygame.font.init()
        # Fallback system fonts with modern clean look
        self.font_family = "consolas,couriernew,menlo,monospace"
        self.font_xs = pygame.font.SysFont(self.font_family, 11)
        self.font_sm = pygame.font.SysFont(self.font_family, 12)
        self.font_base = pygame.font.SysFont(self.font_family, 14)
        self.font_md = pygame.font.SysFont(self.font_family, 16, bold=True)
        self.font_lg = pygame.font.SysFont(self.font_family, 20, bold=True)
        self.font_title = pygame.font.SysFont(self.font_family, 26, bold=True)

    def draw_glass_panel(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        border_color: Tuple[int, int, int] = BORDER_DEFAULT,
        glow: bool = False,
        header_height: int = 0,
        title: str = "",
    ) -> None:
        """Render a dark sci-fi glass panel with translucent fill and neon border."""
        x, y, w, h = rect
        # Translucent fill
        panel_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        panel_surf.fill(BG_PANEL)

        # Header background if present
        if header_height > 0:
            pygame.draw.rect(panel_surf, BG_PANEL_HEADER, (0, 0, w, header_height))
            pygame.draw.line(panel_surf, border_color, (0, header_height), (w, header_height), 1)

        surface.blit(panel_surf, (x, y))

        # Border
        b_color = BORDER_GLOW if glow else border_color
        pygame.draw.rect(surface, b_color, rect, 1)

        # Corner accents (sci-fi bracket look)
        c_len = 6
        pygame.draw.line(surface, BORDER_GLOW, (x, y), (x + c_len, y), 2)
        pygame.draw.line(surface, BORDER_GLOW, (x, y), (x, y + c_len), 2)
        pygame.draw.line(surface, BORDER_GLOW, (x + w, y), (x + w - c_len, y), 2)
        pygame.draw.line(surface, BORDER_GLOW, (x + w, y), (x + w, y + c_len), 2)
        pygame.draw.line(surface, BORDER_GLOW, (x, y + h), (x + c_len, y + h), 2)
        pygame.draw.line(surface, BORDER_GLOW, (x, y + h), (x, y + h - c_len), 2)
        pygame.draw.line(surface, BORDER_GLOW, (x + w, y + h), (x + w - c_len, y + h), 2)
        pygame.draw.line(surface, BORDER_GLOW, (x + w, y + h), (x + w, y + h - c_len), 2)

        # Title text
        if title and header_height > 0:
            txt = self.font_md.render(title.upper(), True, TEXT_PRIMARY)
            surface.blit(txt, (x + 10, y + (header_height - txt.get_height()) // 2))

    def draw_badge(
        self,
        surface: pygame.Surface,
        text: str,
        x: int,
        y: int,
        bg_color: Tuple[int, int, int] = BTN_NORMAL,
        text_color: Tuple[int, int, int] = TEXT_PRIMARY,
        font=None,
    ) -> pygame.Rect:
        font = font or self.font_xs
        txt = font.render(text, True, text_color)
        pad_x, pad_y = 6, 2
        bw = txt.get_width() + pad_x * 2
        bh = txt.get_height() + pad_y * 2
        rect = pygame.Rect(x, y, bw, bh)
        pygame.draw.rect(surface, bg_color, rect, border_radius=3)
        pygame.draw.rect(surface, BORDER_DEFAULT, rect, 1, border_radius=3)
        surface.blit(txt, (x + pad_x, y + pad_y))
        return rect
