import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_learning_curve(episode_rewards):
    plt.clf()
    plt.plot(episode_rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.pause(0.001)

def visualize_q_table(q_table, n_states=5):
    plt.clf()
    sampled_states = list(q_table.keys())[:n_states]
    q_values = np.array([q_table[state] for state in sampled_states])
    sns.heatmap(q_values, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Q-Table Heatmap")
    plt.xlabel("Actions")
    plt.ylabel("Sampled States")
    plt.pause(0.001)