from tensorflow import keras

from core.hex import HexAIGUI
from core.utils import hex_benchmark

model = keras.models.load_model("./hmodel1")

print(f"Mistakes: {hex_benchmark(10000, model)}")

game = HexAIGUI(3, 3, 50, model)
game.draw()
game.loop(ai_first=False)
