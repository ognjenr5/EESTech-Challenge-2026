import sys, numpy as np

from solution import Solution
from logger import ActionLogger
from control import ControlState

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
try:
    import pygame
except ImportError:
    sys.exit("pygame not found. Install with:  pip install pygame")

try:
    from metadrive import MetaDriveEnv
except ImportError:
    sys.exit("MetaDrive not found. Install with:  pip install metadrive-simulator")

# ---------------------------------------------------------------------------
# CONFIG — tune these to feel how you like
# ---------------------------------------------------------------------------
CONFIG = {
    # Control ramping
    "STEER_STEP": 0.05,  # steering added per frame while A/D held
    "STEER_DECAY": 0.07,  # steering reduced per frame toward 0 on release
    "THROTTLE_STEP": 0.05,  # throttle added per frame while W held
    "THROTTLE_DECAY": 0.02,  # throttle reduced per frame toward 0 on W release
    "BRAKE_VALUE": -1,  # throttle value while S held (use 0.0 to just cut)
    "RESET_ON_SPACE": False,  # whether SPACE resets to zero
    # Simulation
    "MAX_STEPS": 10_000,  # safety limit
    "LOG_FILE": "user_input_log.json",
    "ENV_CONFIG": {
        "use_render": True,
        "manual_control": False,
        "traffic_density": 0.3,
        "num_scenarios": 1,
        "start_seed": 0,
        "map": "SOC",  # Straight, rOundabout, Curve
        "accident_prob": 0.0,
        "decision_repeat": 1,
        #"out_of_road_done": False,
        #"crash_vehicle_done": False,
        #"crash_object_done": False
    },
}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

BAR_WIDTH = 40


def _bar(value: float, low: float = -1.0, high: float = 1.0) -> str:
    """Return an ASCII bar centred on zero."""
    fraction = (value - low) / (high - low)  # 0..1
    pos = int(fraction * BAR_WIDTH)
    bar = ["-"] * BAR_WIDTH
    center = BAR_WIDTH // 2
    # mark center
    bar[center] = "|"
    # mark value
    pos = max(0, min(BAR_WIDTH - 1, pos))
    bar[pos] = "#"
    return "[" + "".join(bar) + "]"


def print_status(step: int, steer: float, throttle: float):
    steer_bar = _bar(steer)
    throttle_bar = _bar(throttle)
    print(
        f"\rStep {step:5d} | "
        f"Steer {steer:+.3f} {steer_bar} | "
        f"Throttle {throttle:+.3f} {throttle_bar} | ",
        end="",
        flush=True,
    )


def print_controls():
    print("""
╔══════════════════════════════════════════════════════════════╗
║              MetaDrive Keyboard Controller                   ║
╠══════════════════════════════════════════════════════════════╣
║  A / D    Steer left / right  (auto-center decay on release) ║
║  W        Accelerate          (auto-decay on release)        ║
║  S        Brake / cut throttle (instant 0, stays 0 on rls)   ║
║  SPACE    Reset steering & throttle instantly (if enabled)   ║
║  Q / ESC  Quit and save log                                  ║
╠══════════════════════════════════════════════════════════════╣
║  Steer step   : 0.05 / frame   Steer decay  : 0.07 / frame   ║
║  Throttle step: 0.05 / frame   Throttle dec : 0.02 / frame   ║
╚══════════════════════════════════════════════════════════════╝
""")


def extract_config_from_solution(solution: Solution) -> dict:
    solution_config = solution.config

    result_config = {}

    if "image_observation" in solution_config:
        result_config = {"image_observation": solution_config["image_observation"]}

    if "sensors" in solution_config:
        result_config["sensors"] = solution_config["sensors"]

    vehicle = solution_config.get("vehicle_config", {})
    keys = "image_source"

    vehicle_config = {k: vehicle[k] for k in keys if k in vehicle}

    if vehicle_config:
        result_config["vehicle_config"] = vehicle_config

    return result_config


def get_lidar_observation(env):
    vehicle = env.agent
    lidar_sensor = env.engine.get_sensor("lidar")

    lidar_config = vehicle.config.get("lidar")
    num_lasers = lidar_config.get("num_lasers")
    distance = lidar_config.get("distance")
    show_lidar = vehicle.config.get("show_lidar", False)

    cloud_points, _ = lidar_sensor.perceive(
        vehicle,
        env.engine.physics_world.dynamic_world,
        num_lasers,
        distance,
        show_lidar,
    )

    return np.array(cloud_points)


def sim_out_to_dict(sim_out):
    obs, reward, terminated, truncated, info = sim_out
    return {
        "observation": obs,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


class Game:

    def __init__(self, config=CONFIG, intercepts=[]):
        self.loggers = []
        self.handlers = {}
        self.intercepts = intercepts
        self.config = config

    def start(self):
        print_controls()

        # ------------------------------------------------------------------
        # Init pygame (for keyboard reading only — MetaDrive has its own GL)
        # ------------------------------------------------------------------
        pygame.init()
        # A small window just to capture key events; MetaDrive opens its own.
        pg_screen = pygame.display.set_mode((520, 60))
        pygame.display.set_caption("MetaDrive Controls (focus here for input)")

        # ------------------------------------------------------------------
        # Init helpers
        # ------------------------------------------------------------------
        control = ControlState(self.config)

        step = 0
        running = True

        # ------------------------------------------------------------------
        # Init solution
        # ------------------------------------------------------------------
        solution = Solution(self)

        config = extract_config_from_solution(solution)
        self.config["ENV_CONFIG"].update(config)

        # ------------------------------------------------------------------
        # Init MetaDrive
        # ------------------------------------------------------------------
        print("[MetaDrive] Initializing environment …")
        env = MetaDriveEnv(self.config["ENV_CONFIG"])
        obs, info = env.reset()
        print("[MetaDrive] Environment ready.\n")

        sim_out = {
            "observation": obs,
            "reward": 0,
            "terminated": False,
            "truncated": False,
            "info": info,
        }

        # ------------------------------------------------------------------
        # Main loop
        # ------------------------------------------------------------------
        try:
            while running and step < self.config["MAX_STEPS"]:
                # --- Pygame event pump (must call to keep window responsive) ---
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    # Call custom event handlers if subscribed
                    if event.type in self.handlers:
                        self.handlers[event.type](event)

                keys = pygame.key.get_pressed()

                # Q or ESC → quit
                if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
                    running = False
                    break

                # --- Update control state ---
                steering, throttle = control.update(keys)
                user_input = [steering, throttle]

                # MetaDrive's default observation doesn't include lidar, but some solutions may expect it - so we add it in if needed.
                obs = sim_out["observation"]  # observation dict

                if isinstance(obs, dict) and "lidar" not in obs:
                    obs["lidar"] = get_lidar_observation(env)

                # Possibly intercept or modify the observation before passing to the solution
                for intercept in self.intercepts:
                    obs = intercept(obs)

                sim_out["observation"] = obs

                # --- Call solution iteration with the latest simulator output and user input ---
                next_action = solution.do_iteration(sim_out, user_input=user_input)


                # --- Step the environment ---
                sim_out = sim_out_to_dict(env.step(next_action))

                # --- Log ---
                info = sim_out.get("info", {})

                self._notify_loggers(
                    step,
                    user_steering=steering,
                    user_throttle=throttle,
                    action_steering=float(next_action[0]),
                    action_throttle=float(next_action[1]),
                    reward=float(sim_out.get("reward", 0.0)),
                    terminated=bool(sim_out.get("terminated", False)),
                    truncated=bool(sim_out.get("truncated", False)),
                    speed=info.get("speed", None),
                    velocity=info.get("velocity", None),
                    position=info.get("position", None),
                    crash=info.get("crash", False),
                    out_of_road=info.get("out_of_road", False),
                    arrive_dest=info.get("arrive_dest", False),
                    info=info,
                )

                # --- Display ---
                print_status(step, steering, throttle)

                # Refresh pygame display (tiny window; just keeps it alive)
                pg_screen.fill((20, 20, 20))
                font = pygame.font.SysFont("monospace", 13)
                pg_screen.blit(
                    font.render(
                        f"S:{steering:+.2f}  T:{throttle:+.2f}  step:{step}",
                        True,
                        (200, 230, 200),
                    ),
                    (8, 22),
                )
                pygame.display.flip()

                # --- Episode reset on termination ---
                if sim_out["terminated"] or sim_out["truncated"]:
                    print(f"\n[Episode ended at step {step}] Resetting …")
                    running = False  # Set to False to exit after one episode; set to True to keep going
                    _, _ = env.reset()
                    control.reset()

                step += 1

        except KeyboardInterrupt:
            print("\n[Interrupted by Ctrl+C]")

        finally:
            print("\n[Shutting down …]")
            env.close()
            pygame.quit()
            self._save_loggers()
            self._summarize_loggers()

    def subscribe_logger(self, logger: ActionLogger):
        self.loggers.append(logger)

    def _notify_loggers(self, step: int, **kwargs):
        for logger in self.loggers:
            logger.log(step, **kwargs)

    def _save_loggers(self):
        for logger in self.loggers:
            logger.save()

    def _summarize_loggers(self):
        for logger in self.loggers:
            logger.summary()

    def subscribe_event_handler(self, event_type, handler):
        """Subscribe a custom event handler function that will be called with (event) when the specified pygame event_type occurs.
        Example usage:
            def my_keydown_handler(event):
                if event.key == pygame.K_f:
                    print("F key was pressed!")

            game.subscribe_event_handler(pygame.KEYDOWN, my_keydown_handler)
        """
        self.handlers[event_type] = handler

    def unsubscribe_event_handler(self, event_type):
        """Unsubscribe a previously subscribed event handler for the specified pygame event_type."""
        if event_type in self.handlers:
            del self.handlers[event_type]
