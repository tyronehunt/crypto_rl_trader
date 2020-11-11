import numpy as np
from LinearModel import LinearModel


class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = LinearModel(state_size, action_size)

    def act(self, state):
        # Take in state and choose an action based on that state, using epsilon greedy.
        # If not random, we take the action that leads to max Q-value (argmax over model predictions).
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def train(self, state, action, reward, next_state, done):
        """ Note, our model has multiple outputs, one for each action, but the target is a scalar.
        We also need targets for other outputs (but their error should be zero, or target=prediction).
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)

        # target_full is 2D as num_samples (rows) x num_outputs (cols).
        # In this case: outputs = number of action indexes = 27, samples = 1 as state vector is (1x7)
        # and target_full is (1x27)
        target_full = self.model.predict(state)
        # Only have 1 sample, so row index=0
        target_full[0, action] = target

        # Run one training step (of gradient descent).
        self.model.sgd(state, target_full)

        # Reduce exploration over time.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Load/Save useful to train script with one run, save weights and test with different configurations.
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)