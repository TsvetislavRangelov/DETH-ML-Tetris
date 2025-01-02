import gymnasium as gym
import ale_py
import tetris_agent
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotting
import preprocessor as pp

if __name__ == "__main__":
    learning_rate = 0.1
    n_episodes = 1000
    initial_epsilon = 1.0
    epsilon_decay = initial_epsilon / (n_episodes / 2)
    final_epsilon = 0.1
    gym.register_envs(ale_py)

    env = gym.make("ALE/Tetris-v5")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = tetris_agent.TetrisAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon
    )
    def totuple(a):
        try:
            return tuple(totuple(i) for i in a)
        except TypeError:
            return a
    plt.ion()
    episode_rewards = []
    for episode in tqdm(range(n_episodes)):
        # problem section
        obs = env.reset()
        done = False
        obs = obs[0].flatten()
        obs = pp.Preprocessor.extract_features(obs)
        print(obs)
        total_reward = 0

        while not done: 
            # problem section nr 2
            action = agent.get_action(totuple(obs))
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs_tuple = pp.Preprocessor.extract_features(next_obs[0])
            total_reward += reward

            if terminated or truncated:
                agent.update_qvalues(totuple(obs), action, reward, True, None)
            else:
                agent.update_qvalues(totuple(obs), action, reward, False, totuple(next_obs))
            # Update Q-values with tuples of obs and next_obs
            agent.update_qvalues(obs, action, reward, terminated, next_obs_tuple)
            done = terminated or truncated
            obs = next_obs


        episode_rewards.append(total_reward)
        agent.decay_epsilon()
        # Visualize learning and Q-table
        # if episode % 10 == 0:
        #     plotting.visualize_learning_curve(episode_rewards)
        #     plotting.visualize_q_table(agent.q_values)
        print(total_reward)
    # plt.ioff()
    # plt.show()
    env.close()
    

