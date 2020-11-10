import os


def maybe_make_dir(directory):
    # Checks if a directory exists and creates it if not (to store model and rewards)
    if not os.path.exists(directory):
        os.makedirs(directory)


def play_one_episode(agent, env, is_train):
    """ Reset the environment, get initial state and transform it.
        Use agent to determine each action, perform it in the environment.
        We get back next state, reward, done and info and scale next state.
        Then if in train mode, we train. """
    # note: after transforming states are already 1xD
    state = env.reset()
    state = env.scaler.transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = env.scaler.transform([next_state])
        if is_train == 'train':
            agent.train(state, action, reward, next_state, done)
        state = next_state

    return info['cur_val']
