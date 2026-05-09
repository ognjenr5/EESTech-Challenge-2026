"""
Solution za MetaDrive Autonomous Driving Challenge.

Strategija:
- Koristimo MetaDrive ugradjenu IDMPolicy kao "ekspert autopilot" — ona
  zna da prati put kroz zavoje, kruzne tokove, raskrsnice, sve.
- Korisnik moze da preuzme kontrolu u svakom trenutku (DriverArbiter).
- ADAS slojevi (AEB, ACC, SHM, SmoothSteering) su bezbednosna mreza.

Kljucna stvar: IDMPolicy se instancira tek kad postoji aktivna simulacija
(prvi put kad se pozove do_iteration). Tada pristupamo globalnom engine-u
kroz metadrive API.
"""

from __future__ import annotations

import math

from adas import AdasStack, _clip


class DriverArbiter:
    """Brz attack, spor release — vozac odmah preuzima, autopilot se vraca postepeno."""

    DRIVER_ACTIVE_THRESHOLD = 0.05

    def __init__(self):
        self._driver_intent_steer = 0.0
        self._driver_intent_throttle = 0.0

    def reset(self):
        self._driver_intent_steer = 0.0
        self._driver_intent_throttle = 0.0

    def blend(self, user_steering, user_throttle, ap_steering, ap_throttle):
        target_intent_s = 1.0 if abs(user_steering) > self.DRIVER_ACTIVE_THRESHOLD else 0.0
        target_intent_t = 1.0 if abs(user_throttle) > self.DRIVER_ACTIVE_THRESHOLD else 0.0

        def _smooth(prev, target):
            if target > prev:
                return 0.3 * prev + 0.7 * target
            else:
                return 0.85 * prev + 0.15 * target

        self._driver_intent_steer = _smooth(self._driver_intent_steer, target_intent_s)
        self._driver_intent_throttle = _smooth(self._driver_intent_throttle, target_intent_t)

        w_s = self._driver_intent_steer
        w_t = self._driver_intent_throttle

        steering = w_s * user_steering + (1.0 - w_s) * ap_steering
        throttle = w_t * user_throttle + (1.0 - w_t) * ap_throttle

        return _clip(steering), _clip(throttle)


class ExpertAutopilot:
    """
    Wrapper oko MetaDrive ugradjene IDMPolicy.

    IDMPolicy je provereni ekspert koji prati put kroz sve scenarije
    (zavoji, kruzni tokovi, raskrsnice). Lazy-init kad imamo pristup
    globalnom engine-u.
    """

    def __init__(self):
        self._policy = None
        self._init_attempted = False
        self._init_failed = False
        self._debug_first = True

    def _try_init(self):
        if self._policy is not None or self._init_failed:
            return
        try:
            # MetaDrive ima globalni engine kome se moze pristupiti
            from metadrive.engine.engine_utils import get_engine
            from metadrive.policy.idm_policy import IDMPolicy

            engine = get_engine()
            if engine is None:
                return  # jos uvek nema engine, probacemo kasnije

            # Uzmi prvog (i obicno jedinog) agenta
            agents = list(engine.agents.values()) if hasattr(engine, "agents") else []
            if not agents:
                return

            agent = agents[0]
            seed = getattr(engine, "global_random_seed", 0)
            self._policy = IDMPolicy(agent, seed)
            print(f"[ExpertAutopilot] IDMPolicy inicijalizovan za agenta {agent}")
        except Exception as e:
            if not self._init_attempted:
                print(f"[ExpertAutopilot] Init failed: {type(e).__name__}: {e}")
                print("[ExpertAutopilot] Fallback na neutralan output (vozac vozi sam)")
            self._init_attempted = True
            self._init_failed = True

    def compute(self) -> tuple[float, float]:
        self._try_init()
        if self._policy is None:
            return 0.0, 0.0
        try:
            action = self._policy.act()
            if action is None or len(action) < 2:
                return 0.0, 0.0
            s, t = float(action[0]), float(action[1])
            if self._debug_first:
                print(f"[ExpertAutopilot] Prvi expert output: steer={s:.3f}, throttle={t:.3f}")
                self._debug_first = False
            return _clip(s), _clip(t)
        except Exception as e:
            if self._debug_first:
                print(f"[ExpertAutopilot] act() failed: {type(e).__name__}: {e}")
                self._debug_first = False
            return 0.0, 0.0


class Solution:

    def __init__(self, game):
        self._game = game

        self.expert = ExpertAutopilot()
        self.arbiter = DriverArbiter()
        self.adas = AdasStack()

        self.use_autopilot: bool = True
        self.use_adas: bool = True

        # LKA iskljucujemo jer ekspert vec drzi traku
        self.adas.set_enabled("LKA", False)

        self._step = 0
        self._last_action = (0.0, 0.0)

    @property
    def config(self):
        return {"image_observation": False}

    def do_iteration(self, simulator_output, user_input=None) -> list:
        self._step += 1

        if user_input is None or len(user_input) < 2:
            user_steering, user_throttle = 0.0, 0.0
        else:
            user_steering = _clip(user_input[0])
            user_throttle = _clip(user_input[1])

        if simulator_output is None or "observation" not in simulator_output:
            return [user_steering, user_throttle]

        # 1) Expert autopilot daje predloge
        if self.use_autopilot:
            ap_steering, ap_throttle = self.expert.compute()
        else:
            ap_steering, ap_throttle = 0.0, 0.0

        # 2) Driver arbiter blendje vozaca i autopilota
        if self.use_autopilot:
            base_steer, base_throttle = self.arbiter.blend(
                user_steering, user_throttle, ap_steering, ap_throttle
            )
        else:
            base_steer, base_throttle = user_steering, user_throttle

        # 3) ADAS bezbednosna mreza
        if self.use_adas:
            final_steer, final_throttle = self.adas.process(
                simulator_output, base_steer, base_throttle
            )
        else:
            final_steer, final_throttle = base_steer, base_throttle

        final_steer = _clip(final_steer)
        final_throttle = _clip(final_throttle)

        if not (math.isfinite(final_steer) and math.isfinite(final_throttle)):
            final_steer, final_throttle = 0.0, 0.0

        self._last_action = (final_steer, final_throttle)
        return [final_steer, final_throttle]