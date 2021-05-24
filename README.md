# hexai

We implemented Deep Q-Learning in three environments:

- Simplified Snake game
- Simplified Hex 3x3 game
- Inverse pendulum of OpenAI Gym

### Simplified Snake Game

This game is played in a m-by-n grid in which the player occupies one square.
The goal is to capture the _fruits_ that appear randomly on the grid.
If the player move outwards the grid, they lose the game. No tail is formed, only a single
square is occupied by the player.

Run `toy/toy_script_0` to train the AI to play the toy test game.
The model is saved in `toy/hmodel0` and you can test it by running
`toy/deploy_0`.

Pre-trained models are already available, and you can run the 
deploy scripts directly.

### Hex 3x3 Game

We removed the winning/losing rules of Hex on a 3x3 board and tried to teach
the AI to play only on unoccupied slots on the board. The AI plays on empty slots
alternating with a random player until the board is full.

Run `hex/hex_script_1` to train the AI to play the modified ruleless version
of Hex. The model is saved in `hex/hmodel1` and you can test it by running
`hex/deploy_1`.

Pre-trained models are already available, and you can run the 
deploy scripts directly.

### Inverse Pendulum

We tested our DQN agent in `CartPole-v0` of OpenAI Gym. The current script
is able to learn to balance the pole for over 200 steps, but catastrophic forgetting is happening.

The `gym_script_0` is still WIP, but you can run this script and see that for moments
the agent learns to balance the pole perfeclty, but later forgets.

No pre-trained model is available.
