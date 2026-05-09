"""
ADAS (Advanced Driver Assistance Systems) moduli.

Svaki sloj je nezavisan i može da se uključi/isključi preko `enabled` flega.
Slojevi se izvršavaju u pajplajnu kroz `AdasStack.process()`:

    raw_action  ──►  ACC  ──►  LKA  ──►  CollisionAvoidance  ──►  SmoothSteering  ──►  final_action

Svaki sloj prima trenutnu (steering, throttle) komandu i opciono je modifikuje.
Dizajn liči na realne ADAS arhitekture gde svaki sistem postoji kao zaseban
"asistivni sloj" koji se može testirati i evaluirati nezavisno.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Pomocni moduli
# ---------------------------------------------------------------------------

def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return max(lo, min(hi, float(x)))


def _safe_array(arr) -> np.ndarray:
    """Konvertuje proizvoljan ulaz u 1D numpy array, zamenjujuci NaN/inf."""
    if arr is None:
        return np.array([], dtype=np.float32)
    a = np.asarray(arr, dtype=np.float32).ravel()
    if a.size == 0:
        return a
    a = np.where(np.isfinite(a), a, 1.0)  # NaN/inf -> 1.0 (tj. "prazno")
    return a


# ---------------------------------------------------------------------------
# Perception: izvlacenje korisnih informacija iz raw observation-a
# ---------------------------------------------------------------------------

@dataclass
class Perception:
    """
    Strukturirani pogled na trenutno stanje sveta. Ovo je jedini sloj
    koji "razume" format MetaDrive observation vektora.

    Default LidarStateObservation struktura (ukupno ~259):
        - state[0:9]      ego_state  (steering, heading, speed, side dist, ...)
        - state[9:19]     navigation (2 checkpointa x 5 vrednosti)
        - state[19:259]   lidar      (240 lasera, 0=blizu prepreka, 1=prazno)
    """

    speed_kmh: float = 0.0          # iz info["speed"] kad je dostupno, fallback iz state
    steering_state: float = 0.0     # trenutni steering (state[0])
    heading_diff: float = 0.0       # razlika izmedju heading-a vozila i lane heading-a
    lateral_to_left: float = 0.5    # normalizovana udaljenost do leve ivice trake [0..1]
    lateral_to_right: float = 0.5   # normalizovana udaljenost do desne ivice trake [0..1]

    # Navigacija — predikcija krivine puta unapred
    navi_curvature: float = 0.0     # predznak govori kojim smerom puta krivi
    navi_heading_target: float = 0.0  # ciljani relativni pravac iz navigacije

    # Lidar agregati
    front_min: float = 1.0          # minimalna ocitana distanca u prednjem sektoru [0..1]
    front_left_min: float = 1.0
    front_right_min: float = 1.0
    left_min: float = 1.0
    right_min: float = 1.0
    rear_min: float = 1.0

    # Health
    lidar_valid: bool = True
    obs_valid: bool = True

    raw_lidar: np.ndarray = field(default_factory=lambda: np.ones(240, dtype=np.float32))
    # Necisten lidar — sa NaN-ovima i originalnim vrednostima — za SHM da detektuje degradaciju
    raw_lidar_unfiltered: np.ndarray = field(
        default_factory=lambda: np.ones(240, dtype=np.float32)
    )


class PerceptionExtractor:
    """Pretvara raw simulator observation u strukturirani `Perception` objekat."""

    # Granica state+navi dela u default observation vektoru.
    # Posto MetaDrive moze da menja velicinu state dela u zavisnosti od konfiguracije
    # senzora, NE oslanjamo se na fiksne indekse za sve. Samo na ono sto je stabilno.
    LIDAR_OFFSET_FALLBACK = 19  # default kad nemamo bolju informaciju

    # Ego sektor uglovi (radijani, 0 = napred, raste u smeru kazaljke kod metadrive konvencije)
    # Sektore izrazavamo kao opseg indeksa [low, high) na lidar nizu.
    # Lidar pokriva pun krug 0..2pi sa N lasera; index = angle / (2pi) * N.
    SECTOR_FRONT_HALF_DEG = 25      # +/- ovoliko stepeni je "ispred"
    SECTOR_FRONT_SIDE_DEG = 60      # +/- 60 stepeni je "ispred-strana"
    SECTOR_SIDE_DEG = 30            # +/- 30 stepeni oko 90 i 270 je "bocno"

    def extract(self, sim_out: dict, lidar_array: Optional[np.ndarray] = None) -> Perception:
        p = Perception()

        info = sim_out.get("info", {}) or {}
        obs = sim_out.get("observation", None)

        # --- Brzina iz info-a (najpouzdaniji izvor) ---
        speed = info.get("speed", None)
        if isinstance(speed, (int, float)) and not math.isnan(speed):
            p.speed_kmh = float(speed)

        # --- Lidar i state vektor ---
        state_vec = None
        lidar = None

        if isinstance(obs, dict):
            # Mod gde smo lidar dodali kao zaseban kljuc
            lidar = obs.get("lidar", None)
            state_vec = obs.get("state", None)
        elif isinstance(obs, np.ndarray):
            # Klasican LidarStateObservation: ego_state + navi + lidar u jednom vektoru
            arr = obs.ravel()
            if arr.size > self.LIDAR_OFFSET_FALLBACK:
                state_vec = arr[: self.LIDAR_OFFSET_FALLBACK]
                lidar = arr[self.LIDAR_OFFSET_FALLBACK:]

        if lidar_array is not None and len(lidar_array) > 0:
            lidar = lidar_array

        # --- State parsing (best-effort) ---
        if state_vec is not None:
            sv = _safe_array(state_vec)
            if sv.size >= 9:
                # MetaDrive StateObservation pravilo: prvi ulaz je steering [-1,1] mapirano,
                # ostali su normalizovani [0,1]. Ne zelimo da overengineering-ujemo;
                # uzimamo samo ono sto nam stvarno treba i sto je stabilno.
                # Indeksi: 0=lateral_left, 1=lateral_right, 2=heading_diff, 3=speed_norm,
                # 4=steering, ... (poredak se menja izmedju verzija pa cuvamo robusnost)
                p.lateral_to_left = _clip(sv[0], 0.0, 1.0)
                p.lateral_to_right = _clip(sv[1], 0.0, 1.0)
                # heading_diff je u [0,1] gde je 0.5 == 0 razlike
                p.heading_diff = float(sv[2]) - 0.5 if sv.size > 2 else 0.0
                p.steering_state = _clip(sv[4]) if sv.size > 4 else 0.0
            else:
                p.obs_valid = False

            # Navigacija: 2 checkpointa, svaki ima projekciju x/y na heading,
            # radius, smer i ugao zakrivljenja. Najinformativnije je polje 4 (ugao zavoja
            # za sledeci checkpoint) ali zbog kompatibilnosti uzimamo srednju "udaljenost
            # do levog/desnog ckp" — sto je dovoljno za blagu kurs-korekciju.
            if sv.size >= 19:
                # Procena krivine: levo skretanje predznak '+', desno '-'
                # (pretpostavljamo nav format gde je sv[14] clockwise/anticlockwise flag)
                try:
                    radius1 = float(sv[12])  # radius prvog ckp-a
                    cw1 = float(sv[13])      # 1=clockwise (desno), 0=anticlockwise (levo)
                    if radius1 > 1e-3:
                        p.navi_curvature = (-1.0 if cw1 > 0.5 else 1.0) / max(radius1 * 50.0, 1e-3)
                except Exception:
                    pass

                try:
                    # Tangencijalna projekcija sledeceg checkpointa daje
                    # smer u kom hocemo da idemo
                    p.navi_heading_target = float(sv[10]) - 0.5
                except Exception:
                    pass

        # --- Lidar parsing ---
        # Cuvamo necisten verziju (sa NaN/inf) za SensorHealthMonitor
        if lidar is not None:
            try:
                p.raw_lidar_unfiltered = np.asarray(lidar, dtype=np.float32).ravel()
            except Exception:
                p.raw_lidar_unfiltered = np.array([], dtype=np.float32)

        lidar = _safe_array(lidar)
        if lidar.size < 8:
            # Lidar nije validan ili je premali — degradiran rezim
            p.lidar_valid = False
            p.raw_lidar = np.ones(240, dtype=np.float32)
            return p

        p.raw_lidar = lidar
        n = lidar.size

        # Sektori: indeks 0 je napred. Idemo CW, dakle:
        #   front:   wrap-around oko 0
        #   right:   ~ n/4
        #   rear:    ~ n/2
        #   left:    ~ 3n/4
        # Konvencija u MetaDrive moze da varira, ali simetrije recnice
        # da se ovaj raspored ponasa intuitivno za nas.

        def _sector_min(center_idx: int, half_count: int) -> float:
            """Minimum lidar vrednosti u sektoru centriranom oko `center_idx`."""
            half_count = max(1, half_count)
            # Wrap-around uzimanje
            indices = [(center_idx + k) % n for k in range(-half_count, half_count + 1)]
            vals = lidar[indices]
            # Sigurnost: ako sve nule (sve "u meni" — verovatno bug), vrati 1.0
            if vals.size == 0:
                return 1.0
            return float(np.min(vals))

        deg_per_idx = 360.0 / n

        front_half = max(1, int(self.SECTOR_FRONT_HALF_DEG / deg_per_idx))
        front_side = max(1, int(self.SECTOR_FRONT_SIDE_DEG / deg_per_idx))
        side_half = max(1, int(self.SECTOR_SIDE_DEG / deg_per_idx))

        front_center = 0
        right_center = n // 4
        rear_center = n // 2
        left_center = (3 * n) // 4

        # Front centar
        p.front_min = _sector_min(front_center, front_half)

        # Prednje strane (offset ~30 stepeni od centra)
        offset_30 = max(1, int(30 / deg_per_idx))
        p.front_right_min = _sector_min(front_center - offset_30, front_half)
        p.front_left_min = _sector_min(front_center + offset_30, front_half)

        p.right_min = _sector_min(right_center, side_half)
        p.left_min = _sector_min(left_center, side_half)
        p.rear_min = _sector_min(rear_center, side_half)

        return p


# ---------------------------------------------------------------------------
# ADAS slojevi
# ---------------------------------------------------------------------------

class AdasLayer:
    """Bazna klasa za ADAS sloj. Ima ime, mogucnost ukljucivanja/iskljucivanja."""

    name: str = "AdasLayer"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.last_status: str = ""  # tekstualni status za HCI/dashboard

    def reset(self):
        self.last_status = ""

    def process(
        self,
        steering: float,
        throttle: float,
        perception: Perception,
    ) -> tuple[float, float]:
        if not self.enabled:
            self.last_status = "off"
            return steering, throttle
        return self._apply(steering, throttle, perception)

    def _apply(self, steering, throttle, perception):
        return steering, throttle


# ---------------------------------------------------------------------------
# 1) Lane Keeping Assist
# ---------------------------------------------------------------------------

class LaneKeepingAssist(AdasLayer):
    """
    Drzi vozilo u centru trake. Koristi:
      - lateral pozicija u traci (iz state vektora)
      - heading_diff izmedju vozila i trake
      - navigaciono predvidjanje krivine puta

    Kontroler: PD na lateral grešku + feedforward iz krivine puta.
    """

    name = "LKA"

    def __init__(
        self,
        enabled: bool = True,
        kp_lateral: float = 0.6,
        kp_heading: float = 1.5,
        kp_curvature: float = 1.2,
        max_correction: float = 0.5,
    ):
        super().__init__(enabled)
        self.kp_lateral = kp_lateral
        self.kp_heading = kp_heading
        self.kp_curvature = kp_curvature
        self.max_correction = max_correction

    def _apply(self, steering, throttle, p: Perception):
        # Lateral greska: pozitivno znaci da smo previse desno -> trebamo skrenuti levo
        # Konvencija: lateral_to_left + lateral_to_right ~ 1.0
        lateral_error = p.lateral_to_left - p.lateral_to_right
        # Pozitivno = blize levoj ivici, treba ka desno (negativno steering u metadrive konvenciji,
        # ali kontrole u game.py-u vec namestaju mapping; pa cemo koristiti steering konvenciju
        # gde je + steering = levo skretanje (sto je default MetaDrive konvencija))
        # -> ako smo previse levo (lateral_to_left mali), treba +steering=desno? NE, obrnuto.
        # MetaDrive: + steering = LEVO. Ako smo previse desno (lateral_to_left > lateral_to_right),
        # zelimo +steering (skretanje levo).
        lateral_correction = self.kp_lateral * lateral_error

        heading_correction = -self.kp_heading * p.heading_diff

        curvature_ff = self.kp_curvature * p.navi_curvature

        correction = lateral_correction + heading_correction + curvature_ff
        correction = _clip(correction, -self.max_correction, self.max_correction)

        # Mesamo korisnicki steering sa LKA korekcijom: vozac dominira ako je aktivan,
        # ali LKA daje nezno guranje ka centru trake.
        # Ako vozac aktivno skrece, smanjujemo udeo LKA da ga ne ometamo.
        driver_intent = abs(steering)
        lka_weight = max(0.2, 1.0 - 2.0 * driver_intent)  # 1.0 kad ne skrece, 0.2 kad pun zaokret

        new_steering = _clip(steering + lka_weight * correction)

        self.last_status = f"corr={correction:+.2f}"
        return new_steering, throttle


# ---------------------------------------------------------------------------
# 2) Adaptive Cruise Control
# ---------------------------------------------------------------------------

class AdaptiveCruiseControl(AdasLayer):
    """
    Adaptivno odrzava brzinu i prilagodjava se vozilima/preprekama ispred.

    Ponasanje:
      - Ako je put cist: drzi target_speed (gas u zavisnosti od deficita brzine)
      - Ako je prepreka unutar `following_distance_norm`: smanjuje gas proporcionalno
      - Ako je prepreka ispod `min_safe_distance_norm`: blago koci

    Lidar vrednosti su normalizovane [0,1], gde je 1.0 = nista u opsegu (50m default).
    """

    name = "ACC"

    def __init__(
        self,
        enabled: bool = True,
        target_speed_kmh: float = 45.0,
        following_distance_norm: float = 0.5,    # ~25m (uz 50m lidar)
        min_safe_distance_norm: float = 0.25,    # ~12.5m
        kp_speed: float = 0.05,
    ):
        super().__init__(enabled)
        self.target_speed_kmh = target_speed_kmh
        self.following_distance_norm = following_distance_norm
        self.min_safe_distance_norm = min_safe_distance_norm
        self.kp_speed = kp_speed

    def _apply(self, steering, throttle, p: Perception):
        # Trazena brzina se smanjuje u oštrim zavojima
        curvature_factor = 1.0 - min(0.5, abs(p.navi_curvature) * 5.0)
        effective_target = self.target_speed_kmh * curvature_factor

        speed_error = effective_target - p.speed_kmh

        # Bazna desired throttle za regulisanje brzine
        desired_throttle = self.kp_speed * speed_error
        desired_throttle = _clip(desired_throttle, -0.6, 0.7)

        # Distance-based prigusenje
        front = p.front_min
        if front < self.min_safe_distance_norm:
            # Prepreka jako blizu — pripremi se za jaku redukciju
            desired_throttle = min(desired_throttle, -0.2)
            self.last_status = f"slow d={front:.2f}"
        elif front < self.following_distance_norm:
            # Sledimo na sigurnom rastojanju
            following_factor = (front - self.min_safe_distance_norm) / (
                self.following_distance_norm - self.min_safe_distance_norm
            )
            desired_throttle = min(desired_throttle, 0.5 * following_factor)
            self.last_status = f"follow d={front:.2f}"
        else:
            self.last_status = f"cruise {p.speed_kmh:.0f}"

        # Kombinujemo sa korisnickim ulazom: ACC postavlja "target", a vozac moze
        # da gura jace (kratkotrajno preuzimanje). Uzimamo blendovan output:
        #   - ako je vozac na kocnici (throttle <= -0.5), postujemo ga
        #   - inace blend 50/50 sa preovladjivanjem ka manjoj brzini kad je opasno
        if throttle <= -0.5:
            return steering, throttle  # apsolutni prioritet vozacu kad koci

        # Pristup: ne dozvoljavamo da prelazimo desired_throttle ka gore kad je opasnost
        if front < self.following_distance_norm:
            new_throttle = min(throttle, desired_throttle)
        else:
            # Cist put — blendujemo
            new_throttle = 0.6 * desired_throttle + 0.4 * throttle

        return steering, _clip(new_throttle)


# ---------------------------------------------------------------------------
# 3) Collision Avoidance
# ---------------------------------------------------------------------------

class CollisionAvoidance(AdasLayer):
    """
    Hitan sistem za izbegavanje sudara. Najvisi prioritet od svih ADAS slojeva.

    Detektuje prepreke u prednjem sektoru i:
      1) Ako je prepreka kriticno blizu: hitno koci (throttle = -1)
      2) Ako se moze izbeci skretanjem: blago skreceti u stranu sa vise prostora
      3) Inace: nema akcije
    """

    name = "AEB"  # Autonomous Emergency Braking

    def __init__(
        self,
        enabled: bool = True,
        emergency_distance: float = 0.15,   # ~7.5m
        warning_distance: float = 0.3,      # ~15m
        evasion_steer: float = 0.4,
    ):
        super().__init__(enabled)
        self.emergency_distance = emergency_distance
        self.warning_distance = warning_distance
        self.evasion_steer = evasion_steer

    def _apply(self, steering, throttle, p: Perception):
        front = p.front_min
        front_min_wide = min(p.front_min, p.front_left_min, p.front_right_min)

        if front < self.emergency_distance:
            # KRITIČNO: hitna kočnica + blago skretanje ka strani sa vise prostora
            evasion = 0.0
            if p.front_left_min > p.front_right_min + 0.1 and p.left_min > 0.3:
                evasion = +self.evasion_steer  # u levo
            elif p.front_right_min > p.front_left_min + 0.1 and p.right_min > 0.3:
                evasion = -self.evasion_steer  # u desno

            self.last_status = f"EMERGENCY d={front:.2f}"
            return _clip(steering + evasion), -1.0

        if front_min_wide < self.warning_distance:
            # UPOZORENJE: ogranici gas, ne dozvoli ubrzavanje ka prepreci
            self.last_status = f"warn d={front_min_wide:.2f}"
            return steering, min(throttle, 0.0)

        self.last_status = "clear"
        return steering, throttle


# ---------------------------------------------------------------------------
# 4) Smooth Steering Controller
# ---------------------------------------------------------------------------

class SmoothSteeringController(AdasLayer):
    """
    Slew-rate limiter na komandi steering. Smanjuje jerk i sprečava nagle
    udare na volan koje destabilizuju vozilo.

    Izuzetak: u "emergency" rezimu (vrlo bliska prepreka) puštamo da prolaze
    pune kočnice instantno — bezbednost je iznad glatkoce.
    """

    name = "SSC"

    EMERGENCY_DIST = 0.15  # mora se poklapati sa CollisionAvoidance

    def __init__(
        self,
        enabled: bool = True,
        max_steer_rate: float = 0.08,    # max promena steeringa po koraku
        max_throttle_rate: float = 0.15, # max promena gasa po koraku
        emergency_steer_rate: float = 0.25,
    ):
        super().__init__(enabled)
        self.max_steer_rate = max_steer_rate
        self.max_throttle_rate = max_throttle_rate
        self.emergency_steer_rate = emergency_steer_rate
        self._prev_steer = 0.0
        self._prev_throttle = 0.0

    def reset(self):
        super().reset()
        self._prev_steer = 0.0
        self._prev_throttle = 0.0

    def _apply(self, steering, throttle, p: Perception):
        emergency = p.front_min < self.EMERGENCY_DIST

        # Steering rate limit (vise dozvoljeno u emergency rezimu)
        steer_rate = self.emergency_steer_rate if emergency else self.max_steer_rate
        delta_s = steering - self._prev_steer
        delta_s = _clip(delta_s, -steer_rate, steer_rate)
        new_steering = _clip(self._prev_steer + delta_s)

        # Throttle rate limit
        if emergency and throttle < self._prev_throttle:
            # Hitna kočnica: prolazi instantno, bez rate-limita
            new_throttle = throttle
        elif throttle < self._prev_throttle:
            # Normalno smanjivanje gasa — dozvoljavamo brže nego ubrzavanje
            allowed = max(self.max_throttle_rate * 1.5, 0.2)
            delta_t = _clip(throttle - self._prev_throttle, -allowed, allowed)
            new_throttle = _clip(self._prev_throttle + delta_t)
        else:
            delta_t = _clip(
                throttle - self._prev_throttle,
                -self.max_throttle_rate,
                self.max_throttle_rate,
            )
            new_throttle = _clip(self._prev_throttle + delta_t)

        self._prev_steer = new_steering
        self._prev_throttle = new_throttle

        if emergency:
            self.last_status = f"EMRG d_s={delta_s:+.2f}"
        else:
            self.last_status = f"d_s={delta_s:+.2f}"
        return new_steering, new_throttle


# ---------------------------------------------------------------------------
# 5) Sensor Health Monitor — detektuje degradirane senzore
# ---------------------------------------------------------------------------

class SensorHealthMonitor(AdasLayer):
    """
    Detektuje kada lidar daje nepouzdane podatke (sum, dropout, NaN).
    Kad se to desi:
      - signalizira degradaciju (last_status)
      - preporucuje konzervativnu voznju (smanjenje gasa)
      - ne menja steering (LKA i CA i dalje rade ali na manjoj brzini)

    Implementacija: pratimo (a) udeo nedostajucih ili ekstremnih lidar tackica,
    (b) varijaciju izmedju sukcesivnih ocitavanja (visoka var = sum).
    """

    name = "SHM"

    def __init__(
        self,
        enabled: bool = True,
        dropout_threshold: float = 0.4,    # > 40% dropout-a -> degradacija
        noise_threshold: float = 0.35,     # std razlike izmedju frame-ova
        max_speed_when_degraded: float = 20.0,  # km/h
    ):
        super().__init__(enabled)
        self.dropout_threshold = dropout_threshold
        self.noise_threshold = noise_threshold
        self.max_speed_when_degraded = max_speed_when_degraded
        self._prev_lidar: Optional[np.ndarray] = None
        self.degraded: bool = False
        self.degradation_score: float = 0.0  # [0..1]

    def reset(self):
        super().reset()
        self._prev_lidar = None
        self.degraded = False
        self.degradation_score = 0.0

    def _apply(self, steering, throttle, p: Perception):
        # Koristimo necistu verziju lidara da detektujemo NaN/inf
        lidar_raw = p.raw_lidar_unfiltered
        lidar = p.raw_lidar  # za noise comparison koristimo cistu

        if not p.lidar_valid or lidar.size < 8:
            self.degraded = True
            self.degradation_score = 1.0
            self.last_status = "LIDAR LOST"
            return steering, min(throttle, 0.1)

        # 1) Dropout / stuck-at-zero score (preko necistog lidara!)
        if lidar_raw.size > 0:
            finite_mask = np.isfinite(lidar_raw)
            non_finite_frac = 1.0 - float(finite_mask.mean())
            stuck_at_zero = float((lidar <= 1e-3).mean()) if lidar.size > 0 else 0.0
        else:
            non_finite_frac = 0.0
            stuck_at_zero = 0.0

        dropout_score = min(1.0, non_finite_frac + 2.0 * stuck_at_zero)

        # 2) Noise score: razlika izmedju trenutnog i prethodnog frame-a.
        # Koristimo cistu (filtriranu) verziju da NaN-ovi ne pokvare std.
        noise_score = 0.0
        if (
            self._prev_lidar is not None
            and self._prev_lidar.size == lidar.size
            and lidar.size > 0
        ):
            diff = np.abs(lidar - self._prev_lidar)
            # Median je robusniji na outlier-e (legitimne pojave prepreka)
            noise_score = float(np.median(diff)) * 5.0

        # Ažuriranje pamcenja (sporo, eksponencijalno smoothing)
        if lidar.size > 0:
            if self._prev_lidar is None or self._prev_lidar.size != lidar.size:
                self._prev_lidar = lidar.copy()
            else:
                self._prev_lidar = 0.7 * self._prev_lidar + 0.3 * lidar

        # Kombinovan skor
        score = max(
            dropout_score / max(self.dropout_threshold, 1e-3),
            noise_score / max(self.noise_threshold, 1e-3),
        )
        score = min(1.0, score)
        # Smoothing skora da ne flickeruje
        self.degradation_score = 0.8 * self.degradation_score + 0.2 * score

        if self.degradation_score > 0.7:
            self.degraded = True
            self.last_status = f"DEGRADED {self.degradation_score:.2f}"
            # Konzervativno: ogranici brzinu
            speed_excess = max(0.0, p.speed_kmh - self.max_speed_when_degraded)
            if speed_excess > 0:
                throttle = min(throttle, -0.1)
            else:
                throttle = min(throttle, 0.2)
        elif self.degradation_score > 0.4:
            self.degraded = False
            self.last_status = f"caution {self.degradation_score:.2f}"
            throttle = min(throttle, 0.5)
        else:
            self.degraded = False
            self.last_status = "ok"

        return steering, throttle


# ---------------------------------------------------------------------------
# Stack — orkestrira ADAS slojeve
# ---------------------------------------------------------------------------

class AdasStack:
    """
    Orkestrira ADAS slojeve u definisanom redosledu.

    Redosled je vazan: prvo LKA i ACC modifikuju "normalno" ponasanje,
    onda Collision Avoidance ima poslednju rec o kočnici, zatim
    SmoothSteering izglacava i konacno SHM moze da ogranici gas.
    """

    def __init__(self):
        self.perception_extractor = PerceptionExtractor()
        self.health_monitor = SensorHealthMonitor()
        self.acc = AdaptiveCruiseControl()
        self.lka = LaneKeepingAssist()
        self.collision_avoidance = CollisionAvoidance()
        self.smooth_steering = SmoothSteeringController()

        # Redosled obrade
        self.layers: list[AdasLayer] = [
            self.acc,
            self.lka,
            self.collision_avoidance,
            self.health_monitor,
            self.smooth_steering,  # uvek poslednji da ogranici jerk
        ]

        # Poslednji rezultat (za HCI)
        self.last_perception: Optional[Perception] = None

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def set_enabled(self, layer_name: str, enabled: bool):
        """Ukljuci/iskljuci pojedinacan sloj po imenu."""
        for layer in self.layers:
            if layer.name == layer_name:
                layer.enabled = enabled
                return True
        return False

    def process(
        self,
        sim_out: dict,
        base_steering: float,
        base_throttle: float,
        lidar_array: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        perception = self.perception_extractor.extract(sim_out, lidar_array)
        self.last_perception = perception

        steering = _clip(base_steering)
        throttle = _clip(base_throttle)

        for layer in self.layers:
            steering, throttle = layer.process(steering, throttle, perception)

        return steering, throttle

    def status_summary(self) -> dict:
        """Tekstualni snapshot stanja svakog sloja — za dashboard / debug."""
        return {layer.name: layer.last_status for layer in self.layers}
