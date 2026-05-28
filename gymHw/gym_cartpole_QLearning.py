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

from collections import defaultdict, deque
import gymnasium as gym
import numpy as np

class CartPoleAgent:
    """Initialize simple Q-learning agent for CartPole environment.
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

        self.bins = (
            np.linspace(-2.4, 2.4, 9),
            np.linspace(-3.0, 3.0, 9),
            np.linspace(-0.2095, 0.2095, 9),
            np.linspace(-3.5, 3.5, 9),
        )

    def discretize_state(self, state):
        """Convert CartPole's continuous observation into a hashable Q-table key."""
        clipped_state = np.clip(
            state,
            [-2.4, -3.0, -0.2095, -3.5],
            [2.4, 3.0, 0.2095, 3.5],
        )
        return tuple(
            int(np.digitize(value, state_bins))
            for value, state_bins in zip(clipped_state, self.bins)
        )
    
    def get_action(self, state):
        """Choose an action based on epsilon-greedy strategy."""
        state = self.discretize_state(state)
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return self.env.action_space.sample()
        else:
            # Exploit: choose the best known action
            return int(np.argmax(self.q_values[state]))
        
    def learn(self, state, action, reward, done, next_state):
        # Update Q-value using the Q-learning update rule
        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)
        current_q = self.q_values[state][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_values[next_state])
        self.q_values[state][action] += self.learning_rate * (target_q - current_q)

    def decay_epsilon(self):
        # Decay epsilon after each episode, but don't go below final_epsilon
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        
# Training hyperparameters
learning_rate = 0.1        # How fast to learn (higher = faster but less stable)
n_episodes = 10_000        # Number of episodes to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (2*n_episodes / 5)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration
demo_episodes = 15           # Number of visual episodes to show after training

max_steps_per_episode = 500
success_window = 100
success_threshold = 475
recent_rewards = deque(maxlen=success_window)

def train_agent():
    # Keep render_mode unset during training so learning stays fast.
    env = gym.make("CartPole-v1", max_episode_steps=max_steps_per_episode)

    agent = CartPoleAgent(
        env=env, 
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    # Reset environment to start a new episode
    observation, info = env.reset()
    # observation: cart position, cart velocity, pole angle, pole angular velocity.
    # info: extra debugging information (usually not needed for basic learning)

    print(f"Starting observation: {observation}")
    # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]

    for episode in tqdm(range(n_episodes)):
        # Start a new episode.
        state, info = env.reset()
        episode_over = False
        total_reward = 0

        while not episode_over:
            # Choose an action: 0 = push cart left, 1 = push cart right.
            action = agent.get_action(state)

            # Take the action and see what happens
            next_state, reward, terminated, truncated, info = env.step(action)

            # Learn from next step
            agent.learn(state, action, reward, terminated or truncated, next_state)

            # reward: +1 for each step the pole stays upright.
            # terminated: True if the pole falls or the cart goes out of bounds.
            # truncated: True if we hit the time limit.

            total_reward += reward
            episode_over = terminated or truncated
            state = next_state

        if (episode + 1) % 1_000 == 0:
            print(f"Episode {episode + 1}: reward={total_reward}, epsilon={agent.epsilon:.3f}")
        agent.decay_epsilon()
     
    env.close()
    return agent

# Test the trained agent
def test_agent(agent, env, num_episodes=1000):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

def show_top_q_values(agent, n=200):
    ranked_states = sorted(
        agent.q_values.items(),
        key=lambda item: np.max(item[1]),
        reverse=True,
    )

    print(f"\nTop {n} learned Q-table states:")
    for rank, (state, q_values) in enumerate(ranked_states[:n], start=1):
        best_action = int(np.argmax(q_values))
        best_value = float(np.max(q_values))

        print(
            f"{rank}. state={state}, "
            f"Q(left)={q_values[0]:.3f}, "
            f"Q(right)={q_values[1]:.3f}, "
            f"best_action={best_action}, "
            f"best_value={best_value:.3f}"
        )


def test_agent(agent, delay=0.02):
    demo_env = gym.make(
        "CartPole-v1",
        render_mode="human",
        max_episode_steps=max_steps_per_episode,
    )

    old_env = agent.env
    old_epsilon = agent.epsilon

    agent.env = demo_env
    agent.epsilon = 0.0  # use learned policy, no random exploration

    try:
        state, info = demo_env.reset()
        done = False
        total_reward = 0

        while not done:
            discrete_state = agent.discretize_state(state)
            q_values = agent.q_values[discrete_state]
            action = int(np.argmax(q_values))

            print(
                f"state={discrete_state}, "
                f"Q(left)={q_values[0]:.3f}, "
                f"Q(right)={q_values[1]:.3f}, "
                f"action={action}"
            )

            state, reward, terminated, truncated, info = demo_env.step(action)

            total_reward += reward
            done = terminated or truncated
            time.sleep(delay)

        print(f"Visual demo reward: {total_reward}")
    finally:
        agent.env = old_env
        agent.epsilon = old_epsilon
        demo_env.close()

    return total_reward


def main():
    agent = train_agent()
    show_top_q_values(agent, n=10)
    rewards = []
    for demo in range(demo_episodes):
        print(f"\nVisual sample {demo + 1}/{demo_episodes}")
        reward = test_agent(agent)
        rewards.append(reward)

    print(f"\nTest completed. Max Rewards: {max(rewards)}, Min Rewards: {min(rewards)}, Avg Rewards: {np.mean(rewards):.2f}")

if __name__ == "__main__":
    main()
