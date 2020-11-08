import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler


# Three sets of stock data: AAPL (Apple), MSI (Motorola), SBUX (Starbucks)
def get_data():
    # returns a T x 3 list of (daily close) close prices
    df = pd.read_csv('./data/btc_ohlc_1d.csv')
    return df.values


def get_scaler(env):
    """ Takes a number of steps in the environment, randomly taking actions to get data for scaler.
    :param env: takes an environment object
    :return: scikit-learn scaler object to scale the states
    """
    # Note: you could also populate the replay buffer here.
    # Could run for multiple episodes to improve accuracy.

    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def maybe_make_dir(directory):
    # Checks if a directory exists and creates it if not (to store model and rewards)
    if not os.path.exists(directory):
        os.makedirs(directory)


class LinearModel:
    """ A linear regression model with stochastic gradient descent """

    def __init__(self, input_dim, n_action):
        # input_dim is state dimensionality. n_action is output size, or size of action space.
        # Initialize random weight matrix and bias vector of zeros
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)

        # momentum terms (for W and b respectively)
        self.vW = 0
        self.vb = 0

        # Placeholder for losses on each step of gradient descent
        self.losses = []

    def predict(self, X):
        # Takes in 2D array, X of size N x D
        assert (len(X.shape) == 2)
        return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
        """Does one Step of Gradient Descent: calculate momentum term, v(t). Then update parameters, W, b.
        :param X: training data
        :param Y: target data
        :param learning_rate / momentum: hyper-parameters

        """
        # make sure X is N x D
        assert (len(X.shape) == 2)

        # Our model is linear regression with multiple outputs. Number of samples = N, number of outputs - K.
        # Then y is of size NxK (i.e. loss is 2D). MSE will be divided by num_values.
        num_values = np.prod(Y.shape)

        # do one step of gradient descent
        Yhat = self.predict(X)

        # Calculate gradients.
        # Note d/dx is the gradient of the loss function. i.e. d/dx (x^2) --> 2x. Could incorporate into
        # learning rate, but this is technically correct.
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        # update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        # update params
        self.W += self.vW
        self.b += self.vb

        # Calculate loss for step and append to losses list
        mse = np.mean((Yhat - Y) ** 2)
        self.losses.append(mse)

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)


class MultiStockEnv:
    """
  A 3-stock trading environment.
  State: vector of size 7 (n_stock * 2 + 1)
    - # shares of stock 1 owned
    - # shares of stock 2 owned
    - # shares of stock 3 owned
    - price of stock 1 (using daily close price)
    - price of stock 2
    - price of stock 3
    - cash owned (can be used to purchase more stocks)
  Action: categorical variable with 27 (3^3) possibilities
    - for each stock, you can:
    - 0 = sell
    - 1 = hold
    - 2 = buy
  """

    def __init__(self, data, initial_investment=20000):
        # data
        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape
        print(self.n_step)

        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        self.action_space = np.arange(3 ** self.n_stock)

        # action permutations
        # returns a nested list with elements like:
        # [0,0,0]
        # [0,0,1]
        # [0,0,2]
        # [0,1,0]
        # [0,1,1]
        # etc.
        # 0 = sell
        # 1 = hold
        # 2 = buy
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

        # calculate size of state (as per definition above)
        self.state_dim = self.n_stock * 2 + 1

        self.reset()

    def reset(self):
        # Reset to initial state and return state vector (from _get_obs function)
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()

    def step(self, action):
        """ Perform action in environment and return next state and reward """
        assert action in self.action_space

        # get current value before performing the action
        prev_val = self._get_val()

        # update price, i.e. go to the next day
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]

        # perform the trade
        self._trade(action)

        # get the new value after taking the action
        cur_val = self._get_val()

        # reward is the increase in porfolio value
        reward = cur_val - prev_val

        # done if we have run out of data
        done = self.cur_step == self.n_step - 1

        # store the current value of the portfolio here
        info = {'cur_val': cur_val}

        # conform to the Gym API
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2 * self.n_stock] = self.stock_price
        obs[-1] = self.cash_in_hand
        return obs

    def _get_val(self):
        # Calculates current total value of portfolio
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

    def _trade(self, action):
        # index the action we want to perform
        # 0 = sell
        # 1 = hold
        # 2 = buy
        # e.g. [2,1,0] means:
        # buy first stock
        # hold second stock
        # sell third stock
        action_vec = self.action_list[action]

        # determine which stocks to buy or sell
        sell_index = []  # stores index of stocks we want to sell
        buy_index = []  # stores index of stocks we want to buy
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        # sell any stocks we want to sell
        # then buy any stocks we want to buy
        if sell_index:
            # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
            # NOTE: when buying, we will loop through each stock we want to buy,
            #       and buy one share at a time until we run out of cash
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1  # buy one share
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False


class DQNAgent(object):
    """ """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
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


def play_one_episode(agent, env, is_train):
    """ Reset the environment, get initial state and transform it.
        Use agent to determine each action, perform it in the environment.
        We get back next state, reward, done and info and scale next state.
        Then if in train mode, we train. """
    # note: after transforming states are already 1xD
    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])
        if is_train == 'train':
            agent.train(state, action, reward, next_state, done)
        state = next_state

    return info['cur_val']


if __name__ == '__main__':

    # config
    models_folder = 'linear_rl_trader_models'
    rewards_folder = 'linear_rl_trader_rewards' # from both train / test phases.
    num_episodes = 2000
    batch_size = 32 # for sampling from replay memory
    initial_investment = 20000

    # Enable running the script with command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='--mode can be either "train" or "test"')
    args = parser.parse_args()

    # Create directories, if not already exist.
    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    # Fetch time series
    data = get_data()
    n_timesteps, n_stocks = data.shape

    # Using 50:50 train/test split
    n_train = n_timesteps // 2
    train_data = data[:n_train]
    test_data = data[n_train:]

    # Create training environment and action (with size of state and action space)
    env = MultiStockEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    # store the final value of the portfolio (end of episode)
    portfolio_value = []

    if args.mode == 'test':
        # then load the previous scaler (must be the same as train!)
        with open(f"{models_folder}/scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)

        # remake the env with test data
        env = MultiStockEnv(test_data, initial_investment)

        # make sure epsilon is not default value = 1! (otherwise pure exploration)
        # no need to run multiple episodes if epsilon = 0, it's deterministic
        agent.epsilon = 0.01

        # load trained weights
        agent.load(f'{models_folder}/linear.npz')

    # play the game num_episodes times
    for e in range(num_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent, env, args.mode)
        dt = datetime.now() - t0
        print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
        portfolio_value.append(val)  # append episode end portfolio value

    # save the weights when we are done
    # if args.mode == 'train':
    if args.mode == 'train':
        # save the DQN Agent
        agent.save(f'{models_folder}/linear.npz')

        # save the scaler
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # plot losses
        plt.plot(agent.model.losses)
        plt.show()

    # save list of each of the episodes final portfolio values
    np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)
