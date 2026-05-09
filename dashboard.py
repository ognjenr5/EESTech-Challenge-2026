"""
Dashboard / HCI sloj — vizuelni prikaz stanja ADAS sistema vozacu.

NAPOMENA: Ovaj modul je nezavisan dodatak. Glavna `solution.py` skripta NE
zavisi od njega. Da ga aktivirate, dodajte sledece u svoj main:

    from dashboard import Dashboard
    dashboard = Dashboard(game, solution)
    # game.subscribe_event_handler(...) ili pozovite dashboard.tick() iz petlje

Dizajn po HCI principima:
  1. Hijerarhija — kriticna upozorenja su crvena i veca
  2. Minimalno odvlacenje paznje — samo ono sto je bitno SAD
  3. Konzistentnost — iste boje za iste statuse
  4. Brzo razumevanje — ikone i bar-ovi pre teksta

Boje (svesno odabrane):
  - Zelena: sistem radi, sve OK
  - Žuta: pažnja (warning, blago smanjenje brzine)
  - Crvena: kritično (emergency braking, sensor lost)
  - Plava: informacija (autopilot aktivan)
  - Siva: iskljuceno
"""

from __future__ import annotations

from typing import Optional

try:
    import pygame
except ImportError:
    pygame = None


# Boje (RGB)
COLOR_BG = (15, 18, 25)
COLOR_PANEL = (28, 32, 42)
COLOR_GREEN = (90, 220, 130)
COLOR_YELLOW = (240, 200, 80)
COLOR_RED = (235, 90, 90)
COLOR_BLUE = (110, 170, 240)
COLOR_GRAY = (110, 115, 125)
COLOR_TEXT = (220, 225, 235)
COLOR_TEXT_DIM = (160, 165, 175)


class Dashboard:
    """
    Pygame-based dashboard koji se renderuje u maloj kontrol-window-i.

    Prikazuje:
      - Trenutnu brzinu (veliki broj)
      - Status svakog ADAS sloja (zelena/zuta/crvena lampica)
      - Kriticna upozorenja (preko cele povrsine kad treba)
      - Steering / throttle indikatore

    Cilj: vozac u 100ms vidi sta sistem radi i da li treba da preuzme.
    """

    WIDTH = 520
    HEIGHT = 280  # vise nego default 60 da imamo prostor

    def __init__(self, solution):
        if pygame is None:
            raise RuntimeError("pygame is required for Dashboard")

        self.solution = solution
        self.surface: Optional[pygame.Surface] = None
        self._fonts: dict = {}

    def _font(self, size: int) -> "pygame.font.Font":
        if size not in self._fonts:
            self._fonts[size] = pygame.font.SysFont("monospace", size, bold=False)
        return self._fonts[size]

    def attach(self) -> "pygame.Surface":
        """Pozvati nakon pygame.init(). Vraca surface koji treba da je pg_screen."""
        pygame.display.set_caption("MetaDrive — ADAS Dashboard")
        self.surface = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        return self.surface

    def render(
        self,
        steering: float,
        throttle: float,
        step: int,
        user_steering: float = 0.0,
        user_throttle: float = 0.0,
    ):
        """Pozvati svake iteracije da osveziti prikaz."""
        if self.surface is None:
            return

        s = self.surface
        s.fill(COLOR_BG)

        adas = self.solution.adas
        perception = adas.last_perception
        statuses = adas.status_summary()

        # ----- Header: brzina (najveca info) -----
        speed = perception.speed_kmh if perception else 0.0
        speed_color = COLOR_GREEN
        if speed > 60:
            speed_color = COLOR_YELLOW

        speed_text = self._font(36).render(f"{speed:5.1f}", True, speed_color)
        s.blit(speed_text, (12, 8))
        s.blit(self._font(13).render("km/h", True, COLOR_TEXT_DIM), (130, 26))

        # ----- Mode indicator (autopilot vs manual) -----
        if self.solution.use_autopilot:
            mode_color = COLOR_BLUE
            mode_text = "AUTOPILOT"
        else:
            mode_color = COLOR_GRAY
            mode_text = "MANUAL"
        s.blit(self._font(14).render(mode_text, True, mode_color), (180, 12))
        s.blit(self._font(11).render(f"step {step}", True, COLOR_TEXT_DIM), (180, 30))

        # ----- ADAS layer lights -----
        # Mali "lampici" koji pokazuju status svakog sloja
        layer_y = 60
        layer_x = 12
        layer_specs = [
            ("LKA", "Lane Keep"),
            ("ACC", "Cruise"),
            ("AEB", "Brake"),
            ("SHM", "Sensors"),
            ("SSC", "Smooth"),
        ]
        for name, label in layer_specs:
            # Lampica: cirkulus
            color = self._status_color(name, statuses.get(name, ""), adas)
            pygame.draw.circle(s, color, (layer_x + 8, layer_y + 8), 6)
            # Naziv
            s.blit(self._font(11).render(label, True, COLOR_TEXT), (layer_x + 22, layer_y + 2))
            # Status text (kratak)
            status = statuses.get(name, "")
            if len(status) > 16:
                status = status[:14] + ".."
            s.blit(self._font(10).render(status, True, COLOR_TEXT_DIM), (layer_x + 22, layer_y + 14))
            layer_y += 32

        # ----- Right side: steering & throttle bars -----
        bar_x = 280
        bar_y = 60
        bar_w = 220
        bar_h = 18

        # Steering bar
        s.blit(self._font(11).render("STEERING", True, COLOR_TEXT_DIM), (bar_x, bar_y - 14))
        self._draw_centered_bar(s, bar_x, bar_y, bar_w, bar_h, steering, COLOR_BLUE)

        bar_y += 50
        s.blit(self._font(11).render("THROTTLE", True, COLOR_TEXT_DIM), (bar_x, bar_y - 14))
        # Prikazujemo split: gornja polovina = gas, donja = kocnica
        if throttle >= 0:
            self._draw_centered_bar(s, bar_x, bar_y, bar_w, bar_h, throttle, COLOR_GREEN)
        else:
            self._draw_centered_bar(s, bar_x, bar_y, bar_w, bar_h, throttle, COLOR_RED)

        # User input shadow indicators (manje, ispod)
        bar_y += 40
        s.blit(self._font(10).render("DRIVER", True, COLOR_TEXT_DIM), (bar_x, bar_y - 12))
        ui_w = bar_w // 2 - 8
        self._draw_centered_bar(s, bar_x, bar_y, ui_w, 10, user_steering, COLOR_TEXT_DIM, alpha=0.5)
        self._draw_centered_bar(
            s, bar_x + ui_w + 16, bar_y, ui_w, 10, user_throttle, COLOR_TEXT_DIM, alpha=0.5
        )

        # ----- Critical warnings overlay (spans whole bottom) -----
        warning = self._compute_warning(adas, statuses, perception)
        if warning:
            text, color = warning
            warn_y = self.HEIGHT - 40
            pygame.draw.rect(s, color, (0, warn_y, self.WIDTH, 40))
            warn_text = self._font(20).render(text, True, COLOR_BG)
            text_rect = warn_text.get_rect(center=(self.WIDTH // 2, warn_y + 20))
            s.blit(warn_text, text_rect)

        pygame.display.flip()

    # ------------------------------------------------------------------ helpers

    def _status_color(self, name: str, status: str, adas) -> tuple:
        layer = next((l for l in adas.layers if l.name == name), None)
        if layer is None or not layer.enabled:
            return COLOR_GRAY

        s = status.lower()
        if "emergency" in s or "lost" in s or "degraded" in s:
            return COLOR_RED
        if "warn" in s or "caution" in s or "slow" in s or "follow" in s:
            return COLOR_YELLOW
        return COLOR_GREEN

    def _draw_centered_bar(
        self,
        surface,
        x: int,
        y: int,
        w: int,
        h: int,
        value: float,
        color: tuple,
        alpha: float = 1.0,
    ):
        """Centrirani bar koji se siri iz centra ka levo (negativni val) ili desno (pozitivni)."""
        # Pozadina
        pygame.draw.rect(surface, COLOR_PANEL, (x, y, w, h), border_radius=2)
        # Centralna linija
        center_x = x + w // 2
        pygame.draw.line(surface, COLOR_TEXT_DIM, (center_x, y), (center_x, y + h), 1)

        # Vrednost
        v = max(-1.0, min(1.0, value))
        fill_w = int(abs(v) * (w // 2))
        if v >= 0:
            pygame.draw.rect(surface, color, (center_x, y + 2, fill_w, h - 4), border_radius=2)
        else:
            pygame.draw.rect(
                surface, color, (center_x - fill_w, y + 2, fill_w, h - 4), border_radius=2
            )

    def _compute_warning(self, adas, statuses, perception) -> Optional[tuple]:
        """
        Po HCI principima, prikazujemo SAMO jednu, najvazniju poruku.
        Hijerarhija:
            1. Emergency braking
            2. Sensor lost
            3. Sensor degraded
            4. Slow down (warn)
            5. (nikakva poruka, sistem radi normalno)
        """
        aeb_status = statuses.get("AEB", "")
        if "EMERGENCY" in aeb_status.upper():
            return ("⚠ HITNA KOČNICA", COLOR_RED)

        shm_status = statuses.get("SHM", "")
        if "LOST" in shm_status.upper():
            return ("⚠ SENZOR IZGUBLJEN — RUKE NA VOLAN", COLOR_RED)
        if "DEGRADED" in shm_status.upper():
            return ("✱ Senzor degradiran — voznja oprezno", COLOR_YELLOW)
        if "WARN" in aeb_status.upper():
            return ("✱ Prepreka napred", COLOR_YELLOW)

        return None
