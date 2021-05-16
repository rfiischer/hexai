from tensorflow import keras
from collections import deque
import numpy as np
from random import sample


class DQNCallback(keras.callbacks.Callback):

    loss = ['unavailable']

    def on_epoch_end(self, batch, logs=None):
        self.loss.append(logs["loss"])


class DQNAgent:

    def __init__(self, model, weights, params):

        # Online model
        self.online = keras.models.clone_model(model)
        self.online.set_weights(weights)
        self.online.compile(optimizer=params['optimizer'],
                            loss=params['loss'])

        # Target model
        self.target = keras.models.clone_model(model)
        self.target.set_weights(weights)
        self.target.compile(optimizer=params['optimizer'],
                            loss=params['loss'])

        # Callback
        self.callback = DQNCallback()

        # Replay memory
        self.replay_memory = deque(maxlen=params['memory_size'])

        # Store when to update the target network
        self.online_counter = 0

        # Save parameters
        self.minimum_memory_size = params['minimum_memory_size']
        self.batch_size = params['batch_size']
        self.state_shape = params['state_shape']
        self.action_shape = params['action_shape']
        self.update_target = params['update_target']
        self.discount = params['discount']

    # Append transition to memory
    def update_memory(self, transition):
        self.replay_memory.append(transition)

    # Perform one training step
    def train(self):

        # Wait until there is enough transitions to sample from without probable repetition
        if len(self.replay_memory) >= self.minimum_memory_size:

            # Get a random batch of transitions
            batch = sample(self.replay_memory, self.batch_size)

            # Now we need to compute the target values
            start_states = np.array([transition[0] for transition in batch]).reshape((-1, *self.state_shape))
            start_q = self.online.predict(start_states)

            end_states = np.array([transition[3] for transition in batch]).reshape((-1, *self.state_shape))
            end_q = self.target.predict(end_states)

            # We compute the target values based if we reach a terminal state or not
            x = np.zeros((self.batch_size, *self.state_shape))
            y = start_q
            for idx, (start_state, action, reward, end_state, end_game) in enumerate(batch):

                # If we reach endgame, Q from a future state is 0
                if end_game:
                    target_value = reward

                else:
                    target_value = reward + self.discount * max(end_q[idx, :])

                x[idx, :] = start_state
                y[idx, action] = target_value

            # Now we fit on this minibatch
            self.online.fit(x=x, y=y, batch_size=self.batch_size, verbose=0, shuffle=False,
                            callbacks=[self.callback])

            # Update counter
            self.online_counter += 1

            # Update target network
            if self.online_counter >= self.update_target:
                self.target.set_weights(self.online.get_weights())
                self.online_counter = 0

    # Get the Q values of the online network
    def get_q(self, state):
        return self.online.predict(state.reshape(-1, *self.state_shape))
