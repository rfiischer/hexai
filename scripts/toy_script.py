import numpy as np

from core.games import Toy


# Toy game with boolean states
class BooleanToy(Toy):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.state_shape = (8, )
        self.action_shape = (8, )

    def get_state_from_board(self):
        state = np.zeros((1, 8))

        # Danger zones (close to walls)
        if self.player[1] == 0:
            state[0, 0] = 1

        if self.player[0] == 0:
            state[0, 1] = 1

        if self.player[1] == self.width - 1:
            state[0, 2] = 1

        if self.player[0] == self.height - 1:
            state[0, 3] = 1

        # Where is the reward?
        if self.reward[1] < self.player[1]:
            state[0, 4] = 1

        if self.reward[0] < self.player[0]:
            state[0, 5] = 1

        if self.reward[1] > self.player[1]:
            state[0, 6] = 1

        if self.reward[0] > self.player[0]:
            state[0, 7] = 1

        return state

    @staticmethod
    def get_move_from_action(action):
        return action
