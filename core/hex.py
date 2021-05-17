import numpy as np
from collections import defaultdict
import pygame as pg
import pygame.locals as pl

colors = {1: [255, 0, 0], 2: [0, 0, 255]}


class HexPlayer:
    def __init__(self, x, y, player):
        self.win = False
        self.board = np.zeros((x + 2, y + 2), dtype=np.int8)
        self.groups = defaultdict(list)
        self.available_group = 3
        if player == 'x':
            self.board[0] = 1
            self.board[-1] = 2
        elif player == 'y':
            self.board[:, 0] = 1
            self.board[:, -1] = 2
        else:
            raise Exception()

    def find_surrounding_groups(self, x, y):
        groups = set()
        for _x, _y in [(x, y - 1), (x + 1, y - 1), (x - 1, y), (x + 1, y), (x - 1, y + 1), (x, y + 1)]:
            group = self.board[_x, _y]
            if group != 0:
                groups.add(group)
        return sorted(groups)

    def add_point_to_group(self, x, y, to_group):
        self.groups[to_group].append((x, y))
        self.board[(x, y)] = to_group

    def merge_group(self, from_group, to_group):
        for x, y in self.groups[from_group]:
            self.add_point_to_group(x, y, to_group)

    def place_stone(self, x, y):
        found_groups = self.find_surrounding_groups(x, y)
        if len(found_groups) == 0:
            self.add_point_to_group(x, y, self.available_group)
            self.available_group += 1
        elif found_groups[:2] == [1, 2]:
            self.win = True
        else:
            to_group = found_groups[0]
            self.add_point_to_group(x, y, to_group)
            for from_group in found_groups[1:]:
                self.merge_group(from_group, to_group)


class HexBoard:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.board = np.zeros((x + 2, y + 2), dtype=np.int8)
        self.players = [None, HexPlayer(x, y, 'x'), HexPlayer(x, y, 'y')]  # hacky: 1-based index
        self.board[[0, -1]] = 1
        self.board[:, [0, -1]] = 2
        self.winner = None
        self.crash = False

    def play(self, player_no, x, y):
        current_player = self.players[player_no]
        if self.board[x, y] != 0:
            self.crash = True

        else:
            self.board[x, y] = player_no
            current_player.place_stone(x, y)
            if current_player.win:
                self.winner = player_no

    def get_state(self):
        return self.board[1:-1, 1:-1].copy().reshape((1, self.x, self.y, 1))

    def get_xy(self, position):
        return [position // self.y + 1, position % self.y + 1]

    def get_legal(self):
        return np.where(self.board[1:-1, 1:-1].flatten() == 0)[0]


class HexTextInterface:
    def __init__(self, x, y):
        self.game = HexBoard(x, y)
        self.player_no = 0
        while self.game.winner is None:
            self.loop()
            if self.game.crash:
                print(f"Invalid move, please play again player {self.player_no + 1}")
                self.game.crash = False

            else:
                self.player_no += 1
                self.player_no %= 2

        self.show()
        print(f"Player {self.game.winner} won!")

    def loop(self):
        self.show()
        player_no = self.player_no + 1
        play = input(f"Player #{player_no} to play: ")
        x, y = map(int, play.split(' '))
        self.game.play(player_no, x, y)

    def show(self):
        for k, line in enumerate(self.game.board):
            print(k * ' ', " ".join(str(x) if x != 0 else "." for x in line))


class HexPVPGUI:
    def __init__(self, x, y, size):
        self.x, self.y = x, y
        self.game = HexBoard(x, y)
        self.player_no = 0
        self.square_size = size
        self.rects = dict()
        # Pygame stuff:
        pg.init()
        self.FPS = pg.time.Clock()
        for i in range(y):
            for j in range(x):
                self.rects[j, i] = pg.Rect(
                    (1.1 * (2 * i + j) + 2) * self.square_size / 2,  # x pos
                    (1.1 * j + 1) * self.square_size,  # y pos
                    self.square_size,  # width
                    self.square_size)  # height
        self.display = pg.display.set_mode((
            int(1.1 * (2 * x + y + 2) * self.square_size / 2),
            int(1.1 * (y + 1) * self.square_size)
        ))
        self.display.fill((255, 255, 255))
        pg.display.update()

    def show(self):
        for k, line in enumerate(self.game.board):
            print(k * ' ', " ".join(str(x) if x != 0 else "." for x in line))

    def draw(self):
        top_left = (0.4 * self.square_size, 0.9 * self.square_size)
        top_right = ((1.1 * self.x + 1) * self.square_size, 0.9 * self.square_size)
        bottom_left = ((0.4 + 1.1 * self.y / 2) * self.square_size, (1.1 * self.y + 1) * self.square_size)
        bottom_right = ((0.9 + 1.1 * (self.y / 2 + self.x)) * self.square_size, (1.1 * self.y + 1) * self.square_size)
        pg.draw.line(self.display, colors[1], top_left, top_right, 5)
        pg.draw.line(self.display, colors[1], bottom_left, bottom_right, 5)
        pg.draw.line(self.display, colors[2], top_left, bottom_left, 5)
        pg.draw.line(self.display, colors[2], top_right, bottom_right, 5)
        for rect in self.rects.values():
            pg.draw.rect(self.display, [100, 100, 100], rect)
        pg.display.update()

    def loop(self):
        while True:
            self.FPS.tick(15)
            for event in pg.event.get():
                if event.type == pl.QUIT:
                    pg.display.quit()
                    pg.quit()
                    return

                if event.type == pl.MOUSEBUTTONDOWN and self.game.winner is None:
                    self.mouse_callback(pg.mouse.get_pos())

    def mouse_callback(self, pos):
        for (i, j), rect in self.rects.items():
            if rect.collidepoint(pos):
                current_player_no = self.player_no + 1
                self.game.play(current_player_no, i + 1, j + 1)
                if self.game.crash:
                    print(f"Invalid move, please play again player {self.player_no + 1}")
                    self.game.crash = False

                else:
                    self.player_no += 1
                    self.player_no %= 2
                    pg.draw.rect(self.display, colors[current_player_no], rect)
                    pg.display.update()
                    if self.game.winner is not None:
                        self.cleanup()
                return

    def cleanup(self):
        print(f"Player {self.game.winner} won!")


class HexAIGUI:
    def __init__(self, x, y, size, ai):
        self.x, self.y = x, y
        self.ai = ai
        self.game = HexBoard(x, y)
        self.endgame = False
        self.square_size = size
        self.rects = dict()
        # Pygame stuff:
        pg.init()
        self.FPS = pg.time.Clock()
        for i in range(y):
            for j in range(x):
                self.rects[j, i] = pg.Rect(
                    (1.1 * (2 * i + j) + 2) * self.square_size / 2,  # x pos
                    (1.1 * j + 1) * self.square_size,  # y pos
                    self.square_size,  # width
                    self.square_size)  # height
        self.display = pg.display.set_mode((
            int(1.1 * (2 * x + y + 2) * self.square_size / 2),
            int(1.1 * (y + 1) * self.square_size)
        ))
        self.display.fill((255, 255, 255))
        pg.display.update()

    def show(self):
        for k, line in enumerate(self.game.board):
            print(k * ' ', " ".join(str(x) if x != 0 else "." for x in line))

    def draw(self):
        top_left = (0.4 * self.square_size, 0.9 * self.square_size)
        top_right = ((1.1 * self.x + 1) * self.square_size, 0.9 * self.square_size)
        bottom_left = ((0.4 + 1.1 * self.y / 2) * self.square_size, (1.1 * self.y + 1) * self.square_size)
        bottom_right = ((0.9 + 1.1 * (self.y / 2 + self.x)) * self.square_size, (1.1 * self.y + 1) * self.square_size)
        pg.draw.line(self.display, colors[1], top_left, top_right, 5)
        pg.draw.line(self.display, colors[1], bottom_left, bottom_right, 5)
        pg.draw.line(self.display, colors[2], top_left, bottom_left, 5)
        pg.draw.line(self.display, colors[2], top_right, bottom_right, 5)
        for rect in self.rects.values():
            pg.draw.rect(self.display, [100, 100, 100], rect)
        pg.display.update()

    def loop(self, ai_first=True):

        # AI plays first
        if ai_first:
            self.ai_turn()

        while True:
            self.FPS.tick(15)
            for event in pg.event.get():
                if event.type == pl.QUIT:
                    pg.display.quit()
                    pg.quit()
                    return

                if event.type == pl.MOUSEBUTTONDOWN and not self.endgame:
                    self.mouse_callback(pg.mouse.get_pos())

    def mouse_callback(self, pos):
        for (i, j), rect in self.rects.items():
            if rect.collidepoint(pos):
                # User plays
                move = (i + 1, j + 1)
                self.game.play(2, *move)
                if self.game.crash:
                    print(f"Invalid move, please play again player {2}")
                    self.game.crash = False

                else:
                    pg.draw.rect(self.display, colors[2], self.rects[(i, j)])
                    pg.display.update()
                    if self.game.winner is not None:
                        self.endgame = True
                        print(f"Player {self.game.winner} won!")

                    else:
                        self.ai_turn()

                return

    def ai_turn(self):
        # AI plays
        state = self.game.get_state()
        move = self.game.get_xy(np.argmax(self.ai(state)))
        self.game.play(1, *move)
        if self.game.crash:
            print(f"AI has made a mistake!")
            self.endgame = True

        else:
            pg.draw.rect(self.display, colors[1], self.rects[(move[0] - 1, move[1] - 1)])
            pg.display.update()
            if self.game.winner is not None:
                self.endgame = True
                print(f"Player {self.game.winner} won!")


# To run the GUI, execute the following code
# game = HexPVPGUI(7, 7, 50)
# game.draw()
# game.loop()
