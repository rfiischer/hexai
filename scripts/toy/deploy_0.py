from tensorflow import keras

from core.games import BooleanToy

model = keras.models.load_model("./hmodel0")

toy = BooleanToy(7, 7, model)

toy.ai_loop(fps=10)
