from tensorflow import keras

from core.hex import HexAIGUI

model = keras.models.load_model("./hmodel0")

game = HexAIGUI(3, 3, 50, model.online)
game.draw()
game.loop()
