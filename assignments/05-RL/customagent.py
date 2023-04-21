import numpy as np
import gymnasium as gym


class Agent:
    """Agent class for Lunar Landing"""

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space

    def act_(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        return self.action_space.sample()

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Implements a heuristic action strategy from OpenAI
        """
        s = observation
        angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
        if angle_targ > 0.4:
            angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4:
            angle_targ = -0.4
        hover_targ = 0.55 * np.abs(
            s[0]
        )  # target y should be proportional to horizontal offset

        angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
        hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

        if s[6] or s[7]:  # legs have contact
            angle_todo = 0
            hover_todo = (
                -(s[3]) * 0.5
            )  # override to reduce fall speed, that's all we need after contact
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
        return a

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Passed as we use a heuristic policy

        Args:
            observation (gym.spaces.Box): _description_
            reward (float): _description_
            terminated (bool): _description_
            truncated (bool): _description_
        """
        pass
