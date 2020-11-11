import itertools
import numpy as np
from sklearn.preprocessing import StandardScaler

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
        self.stock_price_history = data[:, [0]]
        self.n_step, self.n_stock = self.stock_price_history.shape
        self.n_hodl = initial_investment / self.stock_price_history[0]

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

        # calculate size of state ( Shape of data, less the price column)
        self.state_dim = data.shape[1] - self.n_stock

        self.reset()

    def reset(self):
        # Reset to initial state and return state vector (from _get_obs function)
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        self.trade_fee = 0.001
        self.min_buy_amount = 200
        return self._get_obs()

    def get_scaler(self):
        """ Takes a number of steps in the environment, randomly taking actions to get data for scaler.
        :param env: takes an environment object
        :return: scikit-learn scaler object to scale the states
        """
        # Note: you could also populate the replay buffer here.
        # Could run for multiple episodes to improve accuracy.

        states = []
        for _ in range(self.n_step):
            action = np.random.choice(self.action_space)
            state, reward, done, info = self.step(action)
            states.append(state)
            if done:
                break

        scaler = StandardScaler()
        scaler.fit(states)
        self.scaler = scaler
        return self.scaler

    def set_scaler(self, scaler_in):
        self.scaler = scaler_in

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

        # sell any stocks we want to sell before we buy any stocks we want to buy
        if sell_index:
            # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i] * (1 - self.trade_fee)
                self.stock_owned[i] = 0
        if buy_index:
            # NOTE: when buying, we will split cash in hand evenly between the purchases
            if self.cash_in_hand > len(buy_index) * self.min_buy_amount:
                buy_amount = np.floor(self.cash_in_hand/len(buy_index))
                for i in buy_index:
                    self.stock_owned[i] += buy_amount * (1 - self.trade_fee) / self.stock_price[i]
                    self.cash_in_hand -= buy_amount