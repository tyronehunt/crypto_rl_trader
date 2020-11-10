import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import argparse
import pickle

from DQNAgent import DQNAgent
from MultiStockEnv import MultiStockEnv
from utils import get_data, maybe_make_dir, play_one_episode


if __name__ == '__main__':

    # config
    cmd_line = False
    args_mode = 'train'

    models_folder = 'linear_rl_trader_models'
    rewards_folder = 'linear_rl_trader_rewards' # from both train / test phases.
    num_episodes = 2000
    batch_size = 32 # for sampling from replay memory
    initial_investment = 20000

    # Enable running the script with command line arguments
    if cmd_line:
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--mode', type=str, required=True,
                            help='--mode can be either "train" or "test"')
        args = parser.parse_args()
        args_mode = args.mode

    # Create directories, if not already exist.
    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    # Fetch time series
    data = get_data('./data/btc_ohlc_1d.csv')
    n_timesteps, n_stocks = data.shape

    # Create train/test split
    train_ratio = 0.8
    n_train = np.floor(n_timesteps * train_ratio).astype(int)
    train_data = data[:n_train]
    test_data = data[n_train:]

    # Create training environment and action (with size of state and action space)
    env = MultiStockEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = env.get_scaler()

    # store the final value of the portfolio (end of episode)
    portfolio_value = []

    if args_mode == 'test':
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
        val = play_one_episode(agent, env, args_mode)
        dt = datetime.now() - t0
        print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
        portfolio_value.append(val)  # append episode end portfolio value

    # save the weights when we are done
    if args_mode == 'train':
        # save the DQN Agent
        agent.save(f'{models_folder}/linear.npz')

        # save the scaler
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # plot losses
        plt.plot(agent.model.losses)
        plt.show()

    # save list of each of the episodes final portfolio values
    np.save(f'{rewards_folder}/{args_mode}.npy', portfolio_value)
