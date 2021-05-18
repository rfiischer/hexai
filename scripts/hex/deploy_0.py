from tensorflow import keras

from core.hex import HexAIGUI
from core.utils import benchmark

model = keras.models.load_model("./hmodel0")

game = HexAIGUI(3, 3, 50, model)
game.draw()
game.loop(ai_first=False)

print(f"Mistakes: {benchmark(10000, model)}")
