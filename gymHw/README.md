# GymHW - Reinforcement Learning Examples

This folder contains reinforcement learning implementations using OpenAI Gymnasium, demonstrating different algorithms and environments.

## Files

### 1. `gym_cartpole_PPO.py`
**Algorithm:** Proximal Policy Optimization (PPO)  
**Environment:** CartPole-v1  
**Description:** Trains a neural network-based agent to balance a pole on a moving cart using the PPO algorithm from `stable_baselines3`.

**Features:**
- Uses MLP (Multi-layer Perceptron) policy
- Trains for 10,000 timesteps
- Displays 4 plots: raw/smoothed rewards and episode lengths
- Runs on CPU

**Dependencies:**
- `stable-baselines3`
- `gymnasium`
- `matplotlib`
- `numpy`

**Run:**
```bash
python gym_cartpole_PPO.py
```

---

### 2. `gym_cartpole_QLearning.py`
**Algorithm:** Q-Learning  
**Environment:** CartPole-v1  
**Description:** Implements a tabular Q-Learning agent from scratch to solve the CartPole problem.

**Features:**
- Manual Q-table management
- Epsilon-greedy exploration strategy
- Customizable learning rate, epsilon decay, and discount factor

**Hyperparameters:**
- Learning rate: 0.1
- Initial epsilon: 1.0
- Discount factor (gamma): 0.99

**Run:**
```bash
python gym_cartpole_QLearning.py
```

---

### 3. `gymnasium_frozenLake_QLearning.py`
**Algorithm:** Q-Learning  
**Environment:** FrozenLake  
**Description:** Solves the FrozenLake environment using Q-Learning, where an agent navigates a frozen lake avoiding holes.

**Features:**
- Simple grid-world problem
- Q-table based approach
- Epsilon-greedy strategy

**Run:**
```bash
python gymnasium_frozenLake_QLearning.py
```

---

## Installation

Install required packages:
```bash
pip install gymnasium stable-baselines3 matplotlib numpy tqdm
```

For classic control environments:
```bash
pip install "gymnasium[classic-control]"
```

---

## Environment Details

- **CartPole-v1:** Agent must balance a pole by moving a cart left/right. Episode succeeds if pole stays upright for 475+ steps.
- **FrozenLake:** Agent navigates a 4×4 frozen lake grid, trying to reach the goal without falling into holes.

---

## Learning Algorithms

- **PPO:** Modern policy gradient method that's stable and sample-efficient
- **Q-Learning:** Classic off-policy temporal difference learning that builds a value table
