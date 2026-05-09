from metadrive import engine


class Solution:

    def __init__(self, game):
        # This reference to the game object may only be used to call its public methods (those not starting with an underscore)
        # If you use this reference to call a method that is not public, it will be considered a violation of the rules and may lead to disqualification
        # If you're unsure whether you can do something with this reference, please ask the organizers for clarification
        self._game = game

    @property
    def config(self):
        # Implement your configuration logic here
        # It is used to define simulator parameters, such as which sensors to use
        # This is just a placeholder implementation


        return {
            "image_observation": False,
            #"sensors": 'lidar'
        }

    def do_iteration(self, simulator_output, user_input=None):

        obs = simulator_output["observation"]
        lidar = obs[19:]

        num_lasers = len(lidar)

        # Prednji sektor (~30 stepeni levo i desno od centra)
        front_sector = lidar[:num_lasers // 12] + lidar[-num_lasers // 12:]

        obstacle_ahead = any(v < 0.5 for v in front_sector)
        obstacle_left = any(
            v < 0.5 for v in lidar[num_lasers // 4 - num_lasers // 12: num_lasers // 4 + num_lasers // 12])
        obstacle_right = any(
            v < 0.5 for v in lidar[-(num_lasers // 4) - num_lasers // 12: -(num_lasers // 4) + num_lasers // 12])

        throttle = 0.75

        if (obstacle_ahead):
            throttle = -1

        user_input = [0, throttle]

        return user_input