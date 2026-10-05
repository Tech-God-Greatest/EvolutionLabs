"""Organism Inspector & Genetic Surgery Workbench (§22, §35)."""
from __future__ import annotations

import json
from typing import Optional
import pygame

from ...analysis.behavior import explain_organism
from ...evolution.genome import (
    G_ATTACK,
    G_METAB,
    G_MUT,
    G_SIZE,
    G_SPEED,
    G_VIS,
    TRAIT_NAMES,
)
from .nn_viz import NetworkVisualizer
from .theme import (
    BORDER_DEFAULT,
    BORDER_GLOW,
    BTN_NORMAL,
    TEXT_ACCENT,
    TEXT_DANGER,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_SUCCESS,
    TEXT_WARNING,
    Theme,
)
from .widgets import Button, Slider


class OrganismInspector:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.active_id: Optional[int] = None
        self.mode = "inspect"  # "inspect" or "surgery" or "family_tree"
        self.nn_viz = NetworkVisualizer(pygame.Rect(rect.x + 10, rect.y + 350, rect.w - 20, 160))

        # Close button
        self.btn_close = Button(pygame.Rect(rect.right - 26, rect.y + 6, 20, 20), "X", self.close)
        # Follow button
        self.btn_follow = Button(pygame.Rect(rect.x + 10, rect.y + 35, 75, 22), "FOLLOW")
        # Genetic Surgery button
        self.btn_surgery = Button(pygame.Rect(rect.x + 95, rect.y + 35, 110, 22), "GENE SURGERY")
        # Export Genome button
        self.btn_export = Button(pygame.Rect(rect.x + 215, rect.y + 35, 70, 22), "EXPORT")

        # Genetic Surgery Sliders
        self.sliders: dict = {}

    def close(self) -> None:
        self.active_id = None
        self.mode = "inspect"

    def set_organism(self, oid: Optional[int], eco) -> None:
        self.active_id = oid
        self.mode = "inspect"
        if oid is not None:
            idx = eco.org.index_of(oid)
            if idx >= 0:
                self._init_surgery_sliders(eco, idx)

    def _init_surgery_sliders(self, eco, idx: int) -> None:
        T = eco.org.traits[idx]
        rx = self.rect.x + 10
        ry = self.rect.y + 70
        w = self.rect.w - 20
        h = 24

        self.sliders = {
            "speed": Slider(pygame.Rect(rx, ry, w, h), "Speed", 15.0, 120.0, float(T[G_SPEED]),
                            lambda v: eco.apply_action("edit_genome", id=self.active_id, traits={"max_speed": v})),
            "vision": Slider(pygame.Rect(rx, ry + 30, w, h), "Vision", 25.0, 200.0, float(T[G_VIS]),
                             lambda v: eco.apply_action("edit_genome", id=self.active_id, traits={"vision_range": v})),
            "metab": Slider(pygame.Rect(rx, ry + 60, w, h), "Metabolism", 0.5, 2.0, float(T[G_METAB]),
                            lambda v: eco.apply_action("edit_genome", id=self.active_id, traits={"metabolism": v})),
            "attack": Slider(pygame.Rect(rx, ry + 90, w, h), "Attack", 0.0, 3.0, float(T[G_ATTACK]),
                             lambda v: eco.apply_action("edit_genome", id=self.active_id, traits={"attack_strength": v})),
            "mutation": Slider(pygame.Rect(rx, ry + 120, w, h), "Mutation", 0.1, 5.0, float(T[G_MUT]),
                               lambda v: eco.apply_action("edit_genome", id=self.active_id, traits={"mutability": v})),
        }

    def handle_event(self, event: pygame.event.Event, eco, camera) -> bool:
        if self.active_id is None:
            return False

        if self.btn_close.handle_event(event):
            return True

        if self.btn_follow.handle_event(event):
            if camera.following_id == self.active_id:
                camera.following_id = None
                self.btn_follow.text = "FOLLOW"
            else:
                camera.following_id = self.active_id
                self.btn_follow.text = "UNFOLLOW"
            return True

        if self.btn_surgery.handle_event(event):
            self.mode = "surgery" if self.mode != "surgery" else "inspect"
            self.btn_surgery.active = (self.mode == "surgery")
            return True

        if self.btn_export.handle_event(event):
            idx = eco.org.index_of(self.active_id)
            if idx >= 0:
                g = eco.genome_of(idx)
                with open(f"saves/organism_{self.active_id}.json", "w", encoding="utf-8") as f:
                    f.write(g.to_json())
                eco.log_event("export", f"Exported genome for VX-{self.active_id:04d}", severity=1)
            return True

        if self.mode == "surgery":
            for slider in self.sliders.values():
                if slider.handle_event(event):
                    return True

        return self.rect.collidepoint(event.pos) if hasattr(event, "pos") else False

    def draw(self, surface: pygame.Surface, eco, camera, theme: Theme) -> None:
        if self.active_id is None:
            return

        idx = eco.org.index_of(self.active_id)
        if idx < 0:
            # Organism perished
            theme.draw_glass_panel(surface, self.rect, BORDER_DEFAULT, header_height=30, title="SPECIMEN EXPIRED")
            self.btn_close.draw(surface, theme)
            txt = theme.font_sm.render(f"Organism #{self.active_id} has died.", True, TEXT_DANGER)
            surface.blit(txt, (self.rect.x + 20, self.rect.y + 60))
            return

        info = eco.organism_info(idx)
        beh = explain_organism(eco, idx)

        # Panel frame
        theme.draw_glass_panel(
            surface,
            self.rect,
            border_color=BORDER_GLOW if self.mode == "surgery" else BORDER_DEFAULT,
            glow=(self.mode == "surgery"),
            header_height=30,
            title=f"SPECIMEN {info['name']} // {info['species']}",
        )

        self.btn_close.draw(surface, theme)
        self.btn_follow.draw(surface, theme)
        self.btn_surgery.draw(surface, theme)
        self.btn_export.draw(surface, theme)

        if self.mode == "surgery":
            # Draw Surgery UI
            stitle = theme.font_md.render("GENETIC MODIFICATION CHAMBER", True, TEXT_WARNING)
            surface.blit(stitle, (self.rect.x + 10, self.rect.y + 65))
            for slider in self.sliders.values():
                slider.draw(surface, theme)
            hint = theme.font_xs.render("Modifications immediately splice into living DNA.", True, TEXT_MUTED)
            surface.blit(hint, (self.rect.x + 10, self.rect.y + 240))
        else:
            # Stats readouts
            y_cur = self.rect.y + 65
            rows = [
                ("Generation", f"{info['generation']}"),
                ("Age", f"{info['age']:.1f} s"),
                ("Energy", f"{info['energy']*100:.1f}%"),
                ("Health", f"{info['health']*100:.1f}%"),
                ("Offspring", f"{info['offspring']}"),
                ("Kills", f"{info['kills']}"),
                ("Fitness Score", f"{info['fitness']:.3f}"),
            ]

            for lbl, val in rows:
                l_txt = theme.font_xs.render(lbl.upper(), True, TEXT_MUTED)
                v_txt = theme.font_xs.render(val, True, TEXT_PRIMARY)
                surface.blit(l_txt, (self.rect.x + 12, y_cur))
                surface.blit(v_txt, (self.rect.x + 130, y_cur))
                y_cur += 18

            # Behavioral Classification Badge & Explanation (§12)
            y_cur += 6
            theme.draw_badge(
                surface,
                f"BEHAVIOR: {beh['name'].upper()} ({beh['confidence']*100:.0f}%)",
                self.rect.x + 12,
                y_cur,
                bg_color=(30, 45, 65),
                text_color=beh["color"],
                font=theme.font_sm,
            )
            y_cur += 26

            # Explainable reasons list
            for reason in beh["reasons"][:3]:
                r_txt = theme.font_xs.render(f"> {reason}", True, TEXT_SECONDARY)
                surface.blit(r_txt, (self.rect.x + 16, y_cur))
                y_cur += 16

        # Neural Network Visualizer
        y_nn = self.rect.y + 355
        lbl_nn = theme.font_xs.render("NEURAL NETWORK TOPOLOGY & SENSORY PATHWAYS", True, TEXT_ACCENT)
        surface.blit(lbl_nn, (self.rect.x + 12, y_nn - 16))
        self.nn_viz.draw(surface, eco, idx, theme)
