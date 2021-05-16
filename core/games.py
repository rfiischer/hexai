import numpy as np
import pygame as pg
from pygame.locals import QUIT, K_UP, K_DOWN, K_LEFT, K_RIGHT, KEYDOWN


class Toy:

    REWARD = 10
    PUNISHMENT = -10

    PLAYER = 1
    FRUIT = 2

    MOVE_TUPLE = ((0, -1), (-1, 0), (0, 1), (1, 0))

    RED = (255, 80, 80)
    GREEN = (80, 255, 80)
    BLACK = (25, 25, 25)
    WHITE = (230, 230, 230)

    ELEMENT_WIDTH = 80
    ELEMENT_HEIGHT = 80
    PADDING = 2

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.display_width = self.width * (self.ELEMENT_WIDTH + self.PADDING) + self.PADDING
        self.display_height = self.height * (self.ELEMENT_HEIGHT + self.PADDING) + self.PADDING
        self.display = None
        self.FPS = None

        self.score = 0
        self.frames_counter = 0
        self.game_over = False

        self.field = np.zeros((self.height, self.width))
        self.player = [np.random.randint(self.height), np.random.randint(self.width)]
        self.reward = None
        self.generate_reward()

        self.place_element(self.player, self.PLAYER)
        self.place_element(self.reward, self.FRUIT)

    def generate_reward(self):
        self.reward = [np.random.randint(self.height), np.random.randint(self.width)]
        while self.reward == self.player:
            self.reward = [np.random.randint(self.height), np.random.randint(self.width)]

        self.place_element(self.reward, self.FRUIT)

    def place_element(self, element, value):
        self.field[tuple(element)] = value

    def move_player(self, move):
        self.field[tuple(self.player)] = 0
        self.player[0] += self.MOVE_TUPLE[move][0]
        self.player[1] += self.MOVE_TUPLE[move][1]
        if 0 <= self.player[0] < self.height and 0 <= self.player[1] < self.width:
            self.field[tuple(self.player)] = self.PLAYER
            valid = True

        else:
            valid = False

        return valid

    def step(self, move):
        valid = self.move_player(move)
        if not valid:
            reward = self.PUNISHMENT
            self.end_game()

        elif self.player == self.reward:
            reward = self.REWARD
            self.generate_reward()
            self.score += self.REWARD
            self.frames_counter += 1

        else:
            self.frames_counter += 1
            reward = 0

        return reward

    def end_game(self):
        self.game_over = True

    def reset(self):
        self.__init__(self.width, self.height)

    def window(self):
        pg.init()
        self.display = pg.display.set_mode((self.display_width,
                                            self.display_height))
        self.display.fill(self.WHITE)
        self.FPS = pg.time.Clock()

    def loop(self):
        self.window()

        while True:
            for event in pg.event.get():
                if event.type == QUIT:
                    pg.display.quit()
                    pg.quit()
                    return

                if event.type == KEYDOWN and not self.game_over:

                    if event.key == K_LEFT:
                        self.step(0)

                    if event.key == K_UP:
                        self.step(1)

                    if event.key == K_RIGHT:
                        self.step(2)

                    if event.key == K_DOWN:
                        self.step(3)

            if not self.game_over:
                self.display.fill(self.WHITE)
                self.draw()

            else:
                self.display.fill(self.BLACK)

            pg.display.update()
            self.FPS.tick(15)

    def draw(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.player == [i, j]:
                    left_top = [j * (self.ELEMENT_WIDTH + self.PADDING) + self.PADDING,
                                i * (self.ELEMENT_HEIGHT + self.PADDING) + self.PADDING]
                    pg.draw.rect(self.display, self.RED, pg.Rect(left_top[0],
                                                                 left_top[1],
                                                                 self.ELEMENT_WIDTH,
                                                                 self.ELEMENT_HEIGHT))

                elif self.reward == [i, j]:
                    left_top = [j * (self.ELEMENT_WIDTH + self.PADDING) + self.PADDING,
                                i * (self.ELEMENT_HEIGHT + self.PADDING) + self.PADDING]
                    pg.draw.rect(self.display, self.GREEN, pg.Rect(left_top[0],
                                                                   left_top[1],
                                                                   self.ELEMENT_WIDTH,
                                                                   self.ELEMENT_HEIGHT))
