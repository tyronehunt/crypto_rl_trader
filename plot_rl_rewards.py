import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True,
                    help='either "train" or "test"')
args = parser.parse_args()

a = np.load(f'linear_rl_trader_rewards/{args.mode}.npy')

print(f"average portfolio: {a.mean():.0f}, min: {a.min():.0f}, max: {a.max():.0f}")

if args.mode == 'train':
  # show the training progress
  plt.plot(a)
  # idx = a < 20000
  # plt.hist([a[idx], a[~idx]], color=['r', 'g'], bins=200)
  # plt.axvline(a.mean(), color='k', linestyle='dashed', linewidth=1)
  # plt.axvline(180000, color='b', linestyle='dashed', linewidth=1)

else:
    # test - show a histogram of final portfolio values
    idx = a < 20000
    plt.hist([a[idx], a[~idx]], color=['r', 'g'], bins=200)
    # Average
    plt.axvline(a.mean(), color='k', linestyle='dashed', linewidth=1)

    # Hodl line
    plt.axvline(14027, color='b', linestyle='dashed', linewidth=1)

plt.title(args.mode)
plt.show()