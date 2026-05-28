from tqdm import tqdm  # Progress bar

from collections import deque
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from matplotlib import pyplot as plt
import numpy as np

device = "cpu"  # Use CPU for training (PPO can be slow on GPU for small environments)

# Training hyperparameters
learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
n_episodes = 10_000        # Number of episodes to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration
demo_episodes = 5           # Number of visual episodes to show after training

max_steps_per_episode = 500
success_window = 100
success_threshold = 475
recent_rewards = []  # Track all episode rewards for plotting
recent_lengths = []  # Track all episode lengths for plotting

def train_agent():
    # Keep render_mode unset during training so learning stays fast.
    env = gym.make("CartPole-v1", max_episode_steps=max_steps_per_episode)

    # 2. Initialize the PPO agent
    # "MlpPolicy" uses a standard neural network (Multi-layer Perceptron)
    agent = PPO("MlpPolicy", env, verbose=1, device=device)

    # 3. Train the agent
    agent.learn(total_timesteps=10000)

    # 4. Save and Test
    agent.save("ppo_cartpole")
    # Reset environment to start a new episode
    observation, info = env.reset()
    # observation: cart position, cart velocity, pole angle, pole angular velocity.
    # info: extra debugging information (usually not needed for basic learning)

    print(f"Starting observation: {observation}")
    # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]

    episode_reward = 0
    episode_length = 0
    for episode in tqdm(range(n_episodes)):
        # Take one step in the episode
        action, _states = agent.predict(observation, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        observation = obs

        if terminated or truncated:
            # Episode ended - record the cumulative reward and length
            recent_rewards.append(episode_reward)
            recent_lengths.append(episode_length)
            episode_reward = 0
            episode_length = 0
            observation, info = env.reset()
     
    env.close()
    return agent

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

def plot_rewards(agent, rewards):

    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(14, 10))

    # 1. Raw episode rewards
    axs[0, 0].set_title("Raw Episode Rewards")
    axs[0, 0].plot(range(len(rewards)), rewards, alpha=0.6, linewidth=0.5)
    axs[0, 0].axhline(y=success_threshold, color='r', linestyle='--', label=f'Success threshold ({success_threshold})')
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].legend()

    # 2. Moving average episode rewards
    axs[0, 1].set_title("Episode Rewards (500-episode Moving Average)")
    reward_moving_average = get_moving_avgs(
        rewards,
        rolling_length,
        "valid"
    )
    axs[0, 1].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0, 1].axhline(y=success_threshold, color='r', linestyle='--', label=f'Success threshold ({success_threshold})')
    axs[0, 1].set_ylabel("Average Reward")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].legend()

    # 3. Raw episode lengths
    axs[1, 0].set_title("Raw Episode Lengths")
    axs[1, 0].plot(range(len(recent_lengths)), recent_lengths, alpha=0.6, linewidth=0.5)
    axs[1, 0].set_ylabel("Episode Length")
    axs[1, 0].set_xlabel("Episode")

    # 4. Moving average episode lengths
    axs[1, 1].set_title("Episode Lengths (500-episode Moving Average)")
    length_moving_average = get_moving_avgs(
        recent_lengths,
        rolling_length,
        "valid"
    )
    axs[1, 1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1, 1].set_ylabel("Average Episode Length")
    axs[1, 1].set_xlabel("Episode")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    agent = train_agent()
    plot_rewards(agent, recent_rewards)