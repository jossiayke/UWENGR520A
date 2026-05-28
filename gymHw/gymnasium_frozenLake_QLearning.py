import subprocess
import sys
import time

# def install_package(package_name):
#     # Runs: python -m pip install <package_name>
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# # Running first RL program within Gymnasium
# # Run `pip install "gymnasium[classic-control]"` for this example.

# install_package("gymnasium[classic-control]")
from tqdm import tqdm  # Progress bar
# try:
# from matplotlib import pyplot as plt
# except ModuleNotFoundError:
#     plt = None

from collections import defaultdict
import gymnasium as gym
import numpy as np

class FrozenLakeAgent:
    """Initialize simple Q-learning agent for FrozenLake environment.
     - Q-table: A dictionary mapping state -> action values (initialized to zero)
     - Learning rate: How quickly the agent updates its knowledge (0 < alpha <= 1)
     - Epsilon: Probability of taking a random action vs. the best known action (exploration vs. exploitation)
     - Discount factor: How much future rewards are worth compared to immediate rewards (0 <= gamma < 1)
    """
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float = 0.1,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 0.01,
            final_epsilon: float = 0.1,
            discount_factor: float = 0.99
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor

        # Q-table: maps state -> action values (initialized to zero)
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        # Tracking learning progress
        self.training_error = []
    
    def get_action(self, state):
        """Choose an action based on epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return self.env.action_space.sample()
        else:
            # Exploit: choose the best known action
            return int(np.argmax(self.q_values[state]))
        
    def learn(self, state, action, reward, done, next_state):
        # Update Q-value using the Q-learning update rule
        current_q = self.q_values[state][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_values[next_state])
        self.q_values[state][action] += self.learning_rate * (target_q - current_q)
        self.training_error.append(target_q - current_q)

    def decay_epsilon(self):
        # Decay epsilon after each episode, but don't go below final_epsilon
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        
# Training hyperparameters
learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
n_episodes = 10_000        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration
demo_episodes = 5           # Number of visual episodes to show after training


def train_agent():
    # Keep render_mode unset during training so pygame is not required.
    env = gym.make("FrozenLake-v1", is_slippery=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = FrozenLakeAgent(
        env=env, 
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    # Reset environment to start a new episode
    observation, info = env.reset()
    # observation: the current tile number on the lake.
    # info: extra debugging information (usually not needed for basic learning)

    print(f"Starting observation: {observation}")
    # Example output: 0

    for episode in tqdm(range(n_episodes)):
        # Start a new episode.
        state, info = env.reset()
        episode_over = False
        total_reward = 0

        while not episode_over:
            # Choose an action: 0 = left, 1 = down, 2 = right, 3 = up.
            action = agent.get_action(state)

            # Take the action and see what happens
            next_state, reward, terminated, truncated, info = env.step(action)

            # Learn from next step
            agent.learn(state, action, reward, terminated or truncated, next_state)

            # reward: +1 for reaching the goal, otherwise 0.
            # terminated: True if the agent reaches a hole or the goal.
            # truncated: True if we hit the environment time limit.

            total_reward += reward
            episode_over = terminated or truncated
            state = next_state

        if (episode + 1) % 1_000 == 0:
            print(f"Episode {episode + 1}: reward={total_reward}, epsilon={agent.epsilon:.3f}")
        agent.decay_epsilon()
    env.close()
    return agent


def run_demo_episode(agent, env, episode, use_text_render=False):
    state, info = env.reset()
    episode_over = False
    total_reward = 0

    while not episode_over:
        action = agent.get_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated

        if use_text_render:
            print(env.render())

        time.sleep(0.3)

    print(f"Demo episode {episode + 1}: reward={total_reward}")


def run_demo(agent, n_demo_episodes=demo_episodes, render_mode="human"):
    # Use a separate environment for visualization after training.
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=render_mode)
    training_env = agent.env
    training_epsilon = agent.epsilon

    agent.env = env
    agent.epsilon = 0.0

    try:
        for episode in range(n_demo_episodes):
            run_demo_episode(agent, env, episode, use_text_render=render_mode == "ansi")
    finally:
        agent.env = training_env
        agent.epsilon = training_epsilon
        env.close()


# def get_moving_avgs(arr, window, convolution_mode):
#     """Compute moving average to smooth noisy data."""
#     return np.convolve(
#         np.array(arr).flatten(),
#         np.ones(window),
#         mode=convolution_mode
#     ) / window

# def plot_training_progress(agent, rolling_length=500):
#     if plt is None:
#         print("Skipping plot: matplotlib is not installed.")
#         print("Install it with: python -m pip install matplotlib")
#         return

#     # Smooth over a 500-episode window
#     rolling_length = 500
#     fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

#     # Episode rewards (win/loss performance)
#     axs[0].set_title("Episode rewards")
#     reward_moving_average = get_moving_avgs(
#         agent.env.return_queue,
#         rolling_length,
#         "valid"
#     )
#     axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
#     axs[0].set_ylabel("Average Reward")
#     axs[0].set_xlabel("Episode")

#     # Episode lengths (how many actions per hand)
#     axs[1].set_title("Episode lengths")
#     length_moving_average = get_moving_avgs(
#         agent.env.length_queue,
#         rolling_length,
#         "valid"
#     )
#     axs[1].plot(range(len(length_moving_average)), length_moving_average)
#     axs[1].set_ylabel("Average Episode Length")
#     axs[1].set_xlabel("Episode")

#     # Training error (how much we're still learning)
#     axs[2].set_title("Training Error")
#     training_error_moving_average = get_moving_avgs(
#         agent.training_error,
#         rolling_length,
#         "same"
#     )
#     axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
#     axs[2].set_ylabel("Temporal Difference Error")
#     axs[2].set_xlabel("Step")

#     plt.tight_layout()
#     plt.show()

def main():
    agent = train_agent()
    # plot_training_progress(agent)
    try:
        run_demo(agent, render_mode="human")
    except Exception as error:
        print(f"Human rendering failed: {error}")
        print("Falling back to text rendering. Fix pygame/libpng to use the GUI renderer.")
        run_demo(agent, render_mode="ansi")

if __name__ == "__main__":
    main()
